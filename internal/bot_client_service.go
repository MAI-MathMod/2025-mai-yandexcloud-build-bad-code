package main

import (
	"log"

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

func HandleUpdate(bot *tgbotapi.BotAPI, event BotEvent) {
	var msg tgbotapi.MessageConfig

	switch event.Text {
	case "/start":
		msg = tgbotapi.NewMessage(event.ChatID, "Здравствуйте! Я являюсь AI ассистентом приёмной комиссии Москвоского Авиационного Института, готов ответить на ваши вопросы, связанные с поступлением в Московский авиационный институт. Чем я могу вам помочь?")
		msg.ReplyMarkup = GetMainKeyboard()
	case "Перевод на оператора":
		msg = tgbotapi.NewMessage(event.ChatID, "Переводим на оператора")
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
