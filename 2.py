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

    

def bdup():
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
    

# Определяем цикл для обработки сообщений
f_toggle: bool = False
while True:
# Получаем сообщения

    longpoll = VkLongPoll(vk_session)
    try:
    # Обрабатываем сообщения с помощью асинхронной функции
        for event in longpoll.listen():
            bdup()
            if event.type == VkEventType.MESSAGE_NEW:
                print(event.text + '   ' + str(event.user_id))
                #vk.messages.send(peer_id=event.peer_id, message='🤖Принято в обработку', random_id=get_random_id())
                try:
                    if event.text[0] != '🤖' and event.text[0] != '❗' and event.text[0] != '⏳' and event.text[0] != '🧔'  and event.text[0] != '🔍' :
                        
                        vk.messages.send(user_id=event.user_id, message='⏳Process...', random_id=get_random_id())  
                except:
                    pass
    except:
        pass
