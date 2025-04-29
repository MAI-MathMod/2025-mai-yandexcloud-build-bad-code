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
			tgbotapi.NewKeyboardButton("FAQ"),
			tgbotapi.NewKeyboardButton("Как пользоваться"),
		),
		tgbotapi.NewKeyboardButtonRow(
			tgbotapi.NewKeyboardButton("Перевод на оператора"),
		),
	)
}

func GetFAQKeyboard() tgbotapi.InlineKeyboardMarkup {
	var rows [][]tgbotapi.InlineKeyboardButton
	for i := 1; i <= 10; i++ {
		btn := tgbotapi.NewInlineKeyboardButtonData(
			"Вопрос "+string(rune('0'+i)),
			"question_"+string(rune('0'+i)),
		)
		rows = append(rows, tgbotapi.NewInlineKeyboardRow(btn))
	}
	return tgbotapi.NewInlineKeyboardMarkup(rows...)
}

func HandleUpdate(bot *tgbotapi.BotAPI, event BotEvent) {
	var msg tgbotapi.MessageConfig

	switch event.Text {
	case "/start":
		msg = tgbotapi.NewMessage(event.ChatID, "Добро пожаловать! Выберите раздел:")
		msg.ReplyMarkup = GetMainKeyboard()
	case "FAQ":
		msg = tgbotapi.NewMessage(event.ChatID, "Выберите вопрос:")
		msg.ReplyMarkup = GetFAQKeyboard()
	case "Как пользоваться":
		msg = tgbotapi.NewMessage(event.ChatID, "Инструкция по использованию сервиса...")
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
