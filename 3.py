import vk_api
import asyncio
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
import requests, json, openai
import multiprocessing, torch, os
from pydub import AudioSegment
from googletrans import Translator
from vk_api.upload import VkUpload
from io import BytesIO
import asyncio, json
from EdgeGPT import Chatbot, ConversationStyle

# Авторизация ВК
token = "token"
vk_session = vk_api.VkApi(token=token)
vk = vk_session.get_api()

# Авторизация Чат ГПТ
keydef = "key"
keyplus = 'key'
openai.api_key = keydef
model_engine = ""

dic = {'key': 'value'}
dic3 = {'key': 'value'}
dic4 = {'key': 'value'}

white_list = ['163435075', '735820189', '201154238', '74334319', '152534192', '209489689']
black_list = ['448409696']

banwrd = ['Режим ответов', 'ChatGPT ENRU', 'Режим чат-бота', 'ChatGPT RU', 'BingAI', 'DALL-E', 'ДвачГПТ', 'Райан Гослинг', 'Бот гопник 18+', 'Сбросить память']

async def bdup():
    global dic, dic3, dic4
    with open('base/data.txt') as file: 
      lines = file.read().splitlines() 
    for line in lines:
      key,value = line.split(': ') 
      dic.update({key:value})

    with open('base/data3.txt') as file: 
      lines = file.read().splitlines() 
    for line in lines:
      key,value = line.split(': ') 
      dic3.update({key:value})

    with open('base/data4.txt') as file: 
      lines = file.read().splitlines() 
    for line in lines:
      key,value = line.split(': ') 
      dic4.update({key:value})

# Формируем запрос к Чат ГПТ
    

# Определяем цикл для обработки сообщени
async def main():
    while True:
    # Получаем сообщения

        longpoll = VkLongPoll(vk_session)
        try:
        # Обрабатываем сообщения с помощью асинхронной функции
            for event in longpoll.listen():
                await bdup()
                if event.type == VkEventType.MESSAGE_NEW:
                    #vk.messages.send(peer_id=event.peer_id, message='🤖Принято в обработку', random_id=get_random_id())
                    try:
                        if event.text not in banwrd:
                            if event.text[0] != '🤖' and event.text[0] != '❗' and event.text[0] != '⏳' and event.text[0] != '🧔'  and event.text[0] != '🔍' :
                                if dic4[str(event.user_id)] == '7':
                                    bot = Chatbot(cookiePath='cookies.json')
                                    answer = await bot.ask(prompt=event.text, conversation_style=ConversationStyle.creative, wss_link="wss://sydney.bing.com/sydney/ChatHub")
                                    print(answer)
                                    try:
                                        bots = answer['item']['messages'][1]['text']
                                        botss = answer['item']['messages'][1]['adaptiveCards'][0]['body'][1]['text'][11:]
                                        botsss = botss.replace(' ', '').replace('](', ' ').replace(')[', ' ').replace(')', '')
                                        src = []
                                        for i in range(len(botsss[1:].split(' '))):
                                            if i%2!=0:
                                                src.append(botsss[1:].split(' ')[i])
                                        for i in range(len(src)):
                                            bots = bots.replace(f'[^{i+1}^]',f'[{src[i]}]')
                                    except:
                                        bots = answer['item']['messages'][1]['text']
                                    try:
                                        vk.messages.edit(
                                            peer_id=event.peer_id,
                                            message_id=event.message_id + 1,
                                            message= '🤖'+bots
                                        )
                                    except:
                                        vk.messages.send(user_id=event.user_id, message= '🤖'+bots,random_id=get_random_id())
                                    await bot.close()
                              
                    except Exception as e:
                        print(e)
                        #vk.messages.send(user_id=event.user_id, message= '❗Сбой генерации, повторите', random_id=get_random_id())
                        pass
        except Exception as e:
            print(e)
            pass

if __name__ == "__main__":
    asyncio.run(main())
