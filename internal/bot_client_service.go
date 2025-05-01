package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

type BotEvent struct {
	ChatID int64
	User   string
	Text   string
}

func GetMainKeyboard() tgbotapi.ReplyKeyboardMarkup {
	return tgbotapi.NewReplyKeyboard(
		tgbotapi.NewKeyboardButtonRow(
			tgbotapi.NewKeyboardButton("Перевод на оператора"),
		),
	)
}

func handleUpdate(bot *tgbotapi.BotAPI, event BotEvent) {
	var msg tgbotapi.MessageConfig

	switch event.Text {
	case "/start":
		msg = tgbotapi.NewMessage(event.ChatID, "Здравствуйте! Я являюсь AI ассистентом приёмной комиссии Московского Авиационного Института, готов ответить на ваши вопросы, связанные с поступлением в Московский авиационный институт. Чем я могу вам помочь?")
		msg.ReplyMarkup = GetMainKeyboard()
	default:
		msg = tgbotapi.NewMessage(event.ChatID, event.Text)
		msg.ReplyMarkup = GetMainKeyboard()
	}

	msg.ParseMode = "Markdown"
	if _, err := bot.Send(msg); err != nil {
		log.Printf("Error sending message: %v", err)
	}
}

func handleOperatorTransfer(event *BotEvent, bot *tgbotapi.BotAPI) {
	link := os.Getenv("OPERATOR_LINK")
	if link == "" {
		msg := tgbotapi.NewMessage(event.ChatID, "⚠️ Оператор временно недоступен")
		bot.Send(msg)
		return
	}

	tgLink := convertToDeepLink(link)

	msg := tgbotapi.NewMessage(
		event.ChatID,
		"Наши специалисты готовы помочь вам в этом чате:",
	)

	msg.ReplyMarkup = tgbotapi.NewInlineKeyboardMarkup(
		tgbotapi.NewInlineKeyboardRow(
			tgbotapi.NewInlineKeyboardButtonURL("Чат с оператором", tgLink),
		),
	)

	if _, err := bot.Send(msg); err != nil {
		log.Printf("Failed to send operator transfer message: %v", err)
	}
}

func convertToDeepLink(originalLink string) string {
	parts := strings.Split(originalLink, "https://t.me/")
	if len(parts) < 2 {
		return originalLink
	}

	username := strings.TrimPrefix(parts[1], "@")
	return fmt.Sprintf("tg://resolve?domain=%s", username)
}
