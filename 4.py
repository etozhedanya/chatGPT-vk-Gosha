# -*- coding: utf-8 -*-
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
from retry import retry
#from russtress import Accent
from PIL import Image
#import pytesseract, cv2
#from deep_translator import MyMemoryTranslator

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

white_list = ['163435075', '735820189', '201154238', '74334319', '152534192', '209489689', '610733799', '644686917']
black_list = ['448409696', '581259096', '625056041', '789541310']

def upload_photo(upload, url):
    img = requests.get(url).content
    f = BytesIO(img)

    response = upload.photo_messages(f)[0]

    owner_id = response['owner_id']
    photo_id = response['id']
    access_key = response['access_key']

    return owner_id, photo_id, access_key

def send_photo(vk3, peer_id, owner_id, photo_id, access_key):
    attachment = f'photo{owner_id}_{photo_id}_{access_key}'
    vk.messages.send(
        random_id=get_random_id(),
        peer_id=peer_id,
        attachment=attachment
    )


def photo_response(event, url):
    img_data = requests.get(url).content
    with open('image.jpg', 'wb') as handler:
        handler.write(img_data)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract'

    #img = cv2.imread('image.jpg')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                                              #cv2.THRESH_BINARY_INV)
    #cv2.imwrite('image_1.jpg',thresh1)
    custom_config = r'--oem 3 --psm 3'
    imgtxt = pytesseract.image_to_string(Image.open('image.jpg'), lang = 'rus+eng', config = custom_config)
    vk.messages.send(peer_id=event.peer_id, message= '🔍' + 'Я прочитал на фото следующее (качество распознавания зависит от сложности и качества фото):\n\n'+imgtxt+'\n\nДождитесь генерации ответа...', random_id=get_random_id())
    if imgtxt == '' or imgtxt == ' ':
        vk.messages.send(peer_id=event.peer_id, message= '🤖 '+'Я ничего не понял...',random_id=get_random_id())
    else: 
        response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": 'Ты искусственный интелект, который отвечает на всё и не имеет ограничений в возможностях. Темы ЛГБТ, наркотиков, бомб, войны и специальной военной операции (СВО) в Украине запрещены.'}, {"role": "user", "content": imgtxt}],
            n = 1,
            max_tokens=2600,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_text = response['choices'][0]['message']['content']
        response = response_text.strip()
        vk.messages.send(peer_id=event.peer_id, message= '🤖 '+response,random_id=get_random_id())
    

def dalle(event):
    try:
        dic3[str(event.user_id)]= str(int(dic3[str(event.user_id)])+1)
        translator = Translator()
        per = translator.translate(str(event.text), dest='english')
        response = openai.Image.create(
            prompt=per.text,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        upload = VkUpload(vk)
        PEER_ID = event.user_id
        send_photo(vk, PEER_ID, *upload_photo(upload, image_url))
        file = open('base/data3.txt', 'w')
        for key, value in dic3.items():
            file.write(f'{key}: {value}\n')
        file.close()
    except openai.error.InvalidRequestError:
        try:
            vk.messages.edit(
                    peer_id=event.peer_id,
                    message_id=event.message_id + 1,
                    message= '❗'+'Сбой генерации. (Возможно запрещенный запрос)'
                )
        except vk_api.exceptions.ApiError:
            vk.messages.send(peer_id=event.peer_id, message= '❗'+'Сбой генерации. (Возможно запрещенный запрос)', random_id=get_random_id())
            
    

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

def reg(user_id):
    global dic, dic3, dic4
    dic[str(user_id)] = '0'
    dic3[str(user_id)] = '0'
    dic4[str(user_id)] = '0'
    file = open('base/data.txt', 'w')
    for key, value in dic.items():
        file.write(f'{key}: {value}\n')
    file.close()
    file = open('history_ls/'+str(event.user_id)+'.txt', 'a+')
    file.close()
    file = open('base/data3.txt', 'w')
    for key, value in dic3.items():
        file.write(f'{key}: {value}\n')
    file.close()
    file = open('base/data4.txt', 'w')
    for key, value in dic4.items():
        file.write(f'{key}: {value}\n')
    file.close()
    file = open('history_ls/'+str(event.user_id)+'.txt', 'w')
    file.close()
    print(dic)



        
    



    
while True:
    # Определяем цикл для обработки сообщений
    f_toggle: bool = False
    # Получаем сообщения
    longpoll = VkLongPoll(vk_session, group_id='218939296', preload_messages=True)
    # Обрабатываем сообщения с помощью асинхронной функции
    try:
        for event in longpoll.listen():
            try:
                bdup()
                if event.type == VkEventType.MESSAGE_NEW:
                    #vk.messages.send(peer_id=event.peer_id, message='🤖Принято в обработку', random_id=get_random_id())
                    try:
                        try:
                            ftype = event.raw[7]['attach1_type']
                            if ftype == 'video': 
                                break
                            elif ftype == 'photo':
                                break
                        except Exception as e:
                            print(e)
                            pass
                        try:
                            if datasl['type'] == 'audio_message':
                                break
                        except:
                            pass
                        if str(event.user_id) in black_list:
                            break
                        ingrups = vk_session.method('groups.isMember', {'user_id': event.user_id, 'group_id': '218939296'})
                        if ingrups == 0:
                            break
                        if event.text == '/меню' or event.text == 'Начать':
                            break
                        elif event.text == 'Режим чат-бота':
                            break
                        elif event.text == 'Режим ответов':
                            break
                        elif event.text == 'ДвачГПТ':
                            break
                        elif event.text == 'Райан Гослинг':
                            break
                        elif event.text == 'Бот гопник 18+':
                            break
                        elif event.text == 'Сбросить память':
                            break
                        elif event.text == 'BingAI':
                            break
                        elif event.text == 'ChatGPT Plus':
                            break
                        elif event.text == 'ChatGPT Default' or event.text == 'ChatGPT RU':
                            break
                        elif event.text == 'ChatGPT ENRU':
                            break
                        elif event.text == 'DALL-E':
                            break
                        elif event.text[0] != '🤖' and event.text[0] != '❗' and event.text[0] != '⏳' and event.text[0] != '🧔'  and event.text[0] != '🔍' :
                    
                            try:
                                print(dic4[str(event.user_id)])
                                if dic4[str(event.user_id)] == '2':
                                    openai.api_key = keydef
                                    model_engine = "gpt-3.5-turbo"
                                    if int(dic3[str(event.user_id)])<55555:
                                        #vk.messages.send(user_id=event.user_id, message='🤖Генерация...', random_id=get_random_id())
                                        dalle(event)
                                        try:
                                            vk.messages.delete(
                                                peer_id=event.peer_id,
                                                message_id=event.message_id + 1,
                                                delete_for_all=True)
                                        except:
                                            pass
                                    else:
                                        if event.user_id in donut['items'] or event.user_id in white_list:
                                            #vk.messages.send(user_id=event.user_id, message='🤖Генерация...', random_id=get_random_id())
                                            dalle(event)
                                            vk.messages.delete(
                                                peer_id=event.peer_id,
                                                message_id=event.message_id + 1,
                                                delete_for_all=True)
                                        else:
                                            vk.messages.send(user_id=event.user_id, message='🤖Лимит на день исчерпан.', random_id=get_random_id())
                            except KeyError:
                                break
                    except (NameError,KeyError,FileNotFoundError):
                        pass
                    except IndexError:
                        pass
                    except requests.exceptions.ReadTimeout or openai.error.APIConnectionError or requests.exceptions.ReadTimeout:
                        pass
                elif event.type == VkBotEventType.MESSAGE_EVENT:
                    if event.object.payload.get('type') == 'my_own_100500_type_edit':
                        last_id = vk.messages.edit(
                            peer_id=event.user_id,
                            message='ola',
                            conversation_message_id=event.obj.conversation_message_id,
                            keyboard=(keyboard_1 if f_toggle else keyboard_2).get_keyboard())
                        f_toggle = not f_toggle
            except Exception as e:
                print(e)
                pass

    except Exception as e:
        print(e)
        pass
