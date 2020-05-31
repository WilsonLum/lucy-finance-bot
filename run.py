# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:19:05 2020

@author: Donal
"""
import telegram
from telegram.ext import Updater
import logging
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
import utils.utterance as utterance
from Training import training, attentiontraining
from Training.attentiontraining import Utterances, Labels


Utterances("utterance")
Labels("label")
# Command Handler
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('How are you ? I am Lucy. '
                              'I can help you with Technology Stock price direction predictions '
                              'and the stock news itself or any trending technology stock news or topics. '
                              'OK, To get started, let me know which stock are you interested in ? '
                              'To end the conversation just say Bye. ')


# Chat Handler
def chat(update, context):
    # reply = utterance.getreply(update.message.text)
    reply = utterance.getreply(update, context)
    print("User text : ", update.message.text)
    print("Reply to user:" + str(reply))
    context.bot.send_message(chat_id=update.effective_chat.id, text=reply["Reply"]
                             , parse_mode=telegram.ParseMode.HTML)

def changedate(update, context):
    userdata = context.user_data
    userdata["PI"] = "date"
    context.bot.send_message(chat_id=update.effective_chat.id, text="which date?"
                             , parse_mode=telegram.ParseMode.HTML)
if __name__ == '__main__':
    training.train()
    #attentiontraining.train()
    try:
        with open('z.token key.txt', 'r') as file:
            TokenKey = file.read().replace('\n', '')
    except "invalid key":
        TokenKey = "<your token key here>"
        if TokenKey == "<your token key here>":
            print('invalid token key')

    updater = Updater(token=TokenKey, use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("date", changedate))

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    dispatcher.add_handler(MessageHandler(Filters.text, chat))

    updater.start_polling()

    updater.idle()
