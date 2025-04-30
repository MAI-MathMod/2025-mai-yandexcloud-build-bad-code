package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/sqs"
	sqstypes "github.com/aws/aws-sdk-go-v2/service/sqs/types"
)

const (
	batchSize      = 10
	batchWait      = 100 * time.Millisecond
	receiveWorkers = 5
)

var (
	sqsClient       *sqs.Client
	botQueueURL     string
	ragResponseURL  string
	botQueueName    = "bot-events.fifo"
	ragResponseName = "RAG_response"
	batchBuf        chan *sqs.SendMessageInput
)

func main() {
	token := os.Getenv("BOT_TOKEN")
	if token == "" {
		log.Fatal("BOT_TOKEN not set")
	}

	bot, err := tgbotapi.NewBotAPI(token)
	if err != nil {
		log.Panic(err)
	}
	log.Printf("Authorized as %s", bot.Self.UserName)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	initAWS(ctx)
	defer closeAWS(ctx)

	batchBuf = make(chan *sqs.SendMessageInput, 1000)
	initBatchSender(ctx)

	startConsumers(ctx, bot)

	u := tgbotapi.NewUpdate(0)
	u.Timeout = 30
	updates := bot.GetUpdatesChan(u)

	for {
		select {
		case <-ctx.Done():
			log.Println("Shutting down main loop")
			return
		case update, ok := <-updates:
			if !ok {
				log.Println("Updates channel closed")
				stop()
				continue
			}
			if update.Message == nil {
				continue
			}

			if isMediaMessage(update.Message) {
				msg := tgbotapi.NewMessage(
					update.Message.Chat.ID,
					"⚠️ Я принимаю только текстовые сообщения. Пожалуйста, отправьте текст.",
				)
				bot.Send(msg)
				continue
			}

			username := "Anonymous"
			if update.Message.From != nil && update.Message.From.UserName != "" {
				username = update.Message.From.UserName
			}

			ev := &BotEvent{
				ChatID: update.Message.Chat.ID,
				Text:   update.Message.Text,
			}

			if checkCommand(ev, bot) {
				continue
			}

			input := &sqs.SendMessageInput{
				QueueUrl:       &botQueueURL,
				MessageBody:    aws.String(ev.Text),
				MessageGroupId: aws.String(strconv.FormatInt(ev.ChatID, 10)),
				MessageAttributes: map[string]sqstypes.MessageAttributeValue{
					"ChatID": {
						DataType:    aws.String("Number"),
						StringValue: aws.String(strconv.FormatInt(ev.ChatID, 10)),
					},
					"Username": {
						DataType:    aws.String("String"),
						StringValue: aws.String(username),
					},
				},
			}

			select {
			case batchBuf <- input:
			default:
				log.Println("Batch buffer full, dropped event")
			}
		}
	}
}

func initAWS(ctx context.Context) {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConnsPerHost = 100

	resolver := aws.EndpointResolverWithOptionsFunc(func(service, region string, opts ...interface{}) (aws.Endpoint, error) {
		return aws.Endpoint{
			URL:           "https://message-queue.api.cloud.yandex.net",
			SigningRegion: "ru-central1",
		}, nil
	})

	creds := credentials.NewStaticCredentialsProvider(
		os.Getenv("YC_SERVICE_ACCESS_ID"),
		os.Getenv("YC_ACCESS_KEY"),
		"",
	)

	cfg, err := config.LoadDefaultConfig(
		ctx,
		config.WithCredentialsProvider(creds),
		config.WithEndpointResolverWithOptions(resolver),
		config.WithHTTPClient(&http.Client{Transport: tr}),
	)
	if err != nil {
		log.Fatalf("failed to load AWS config: %v", err)
	}

	sqsClient = sqs.NewFromConfig(cfg)

	botOut, err := sqsClient.CreateQueue(ctx, &sqs.CreateQueueInput{
		QueueName: &botQueueName,
		Attributes: map[string]string{
			"FifoQueue":                 "true",
			"ContentBasedDeduplication": "true",
		},
	})
	if err != nil {
		log.Fatalf("failed to create bot-events queue: %v", err)
	}
	botQueueURL = *botOut.QueueUrl
	log.Printf("Bot-events Queue URL: %s", botQueueURL)

	ragOut, err := sqsClient.CreateQueue(ctx, &sqs.CreateQueueInput{
		QueueName: &ragResponseName,
	})
	if err != nil {
		log.Fatalf("failed to create RAG_response queue: %v", err)
	}
	ragResponseURL = *ragOut.QueueUrl
	log.Printf("RAG_response Queue URL: %s", ragResponseURL)
}

func initBatchSender(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(batchWait)
		defer ticker.Stop()

		var batch []sqstypes.SendMessageBatchRequestEntry
		flush := func() {
			if len(batch) == 0 {
				return
			}
			_, err := sqsClient.SendMessageBatch(ctx, &sqs.SendMessageBatchInput{
				QueueUrl: &botQueueURL,
				Entries:  batch,
			})
			if err != nil {
				log.Printf("Batch send error: %v", err)
			}
			batch = batch[:0]
		}

		for {
			select {
			case <-ctx.Done():
				flush()
				return
			case in := <-batchBuf:
				batch = append(batch, sqstypes.SendMessageBatchRequestEntry{
					Id:                aws.String(strconv.Itoa(len(batch))),
					MessageBody:       in.MessageBody,
					MessageAttributes: in.MessageAttributes,
					MessageGroupId:    in.MessageGroupId,
				})
				if len(batch) >= batchSize {
					flush()
				}
			case <-ticker.C:
				flush()
			}
		}
	}()
}

func startConsumers(ctx context.Context, bot *tgbotapi.BotAPI) {
	for i := 0; i < receiveWorkers; i++ {
		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}

				out, err := sqsClient.ReceiveMessage(ctx, &sqs.ReceiveMessageInput{
					QueueUrl:              &ragResponseURL,
					MaxNumberOfMessages:   10,
					WaitTimeSeconds:       20,
					MessageAttributeNames: []string{"All"},
				})
				if err != nil {
					log.Printf("Receive error: %v", err)
					continue
				}

				for _, msg := range out.Messages {
					go func(m sqstypes.Message) {
						attr := m.MessageAttributes["ChatID"]
						chatID, _ := strconv.ParseInt(*attr.StringValue, 10, 64)
						HandleUpdate(bot, BotEvent{ChatID: chatID, Text: *m.Body})

						_, err := sqsClient.DeleteMessage(ctx, &sqs.DeleteMessageInput{
							QueueUrl:      &ragResponseURL,
							ReceiptHandle: m.ReceiptHandle,
						})
						if err != nil {
							log.Printf("Delete error: %v", err)
						}
					}(msg)
				}
			}
		}()
	}
}

func isMediaMessage(msg *tgbotapi.Message) bool {
	return msg.Photo != nil ||
		msg.Video != nil ||
		msg.Document != nil ||
		msg.Audio != nil ||
		msg.Voice != nil ||
		msg.VideoNote != nil ||
		msg.Sticker != nil ||
		msg.Location != nil
}

func checkCommand(event *BotEvent, bot *tgbotapi.BotAPI) bool {
	switch event.Text {
	case "/start", "Перевод на оператора":
		HandleUpdate(bot, *event)
		return true
	}
	return false
}

func closeAWS(ctx context.Context) {
	_, err := sqsClient.DeleteQueue(ctx, &sqs.DeleteQueueInput{QueueUrl: &botQueueURL})
	if err != nil {
		log.Printf("Error deleting bot-events queue: %v", err)
	}
	_, err = sqsClient.DeleteQueue(ctx, &sqs.DeleteQueueInput{QueueUrl: &ragResponseURL})
	if err != nil {
		log.Printf("Error deleting RAG_response queue: %v", err)
	}
}
