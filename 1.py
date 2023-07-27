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
from russtress import Accent
from PIL import Image
import pytesseract, cv2
from deep_translator import MyMemoryTranslator

# Авторизация ВК
token = "vk.token"
vk_session = vk_api.VkApi(token=token)
vk = vk_session.get_api()

# Авторизация Чат ГПТ
keydef = "chat-gpt-key"
keyplus = 'chat-gpt-key-plus'
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
            
        

def govor(response):
    device = torch.device('cpu')

    torch.set_num_threads(4)
    local_file = 'model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                       local_file)
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    example_text = response
    sample_rate = 48000
    speaker='aidar'
    audio_paths = model.save_wav(text=str(example_text),speaker=speaker,sample_rate=sample_rate)

def whisper(event, url):
    global dic, dic3, dic4
    history = open('history_ls/'+str(event.user_id)+'.txt', 'r')
    conversation_history = history.read()
    history.close()
    user_id = event.user_id
    r = requests.get(url)
    with open('gs.mp3', 'wb') as f:
        f.write(r.content)
    audio_file= open("gs.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language = 'ru')
    if dic[str(event.user_id)] == '0':
        response = gptresponsex(transcript.text, user_id)
    else:
        response = gptresponse(transcript.text, conversation_history, user_id)
        conversation_history2 = f"Пользователь: {transcript.text}\nЧат-бот: {response}\n"
        try:
            history = open('history_ls/'+str(event.user_id)+'.txt', 'a+')
            history.write(conversation_history2)
            history.close()
        except UnicodeEncodeError:
            pass
    govor(response)
    given_audio = AudioSegment.from_file("test.wav", format="wav")
    given_audio.export("gsh.mp3", format="mp3")
    upload = VkUpload(vk)
    audio_msg = upload.audio_message('gsh.mp3', peer_id=event.peer_id)
    attachment = 'doc{}_{}'.format(audio_msg['audio_message']['owner_id'], audio_msg['audio_message']['id'])
    vk.messages.send(peer_id=event.peer_id, attachment=attachment, random_id=get_random_id())
    

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

def create_keyboard():
    keyboard = VkKeyboard(one_time=False)
    #False Если клавиатура должна оставаться откртой после нажатия на кнопку
    #True если она должна закрваться
    keyboard.add_button("Режим ответов", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("ChatGPT ENRU", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Режим чат-бота", color=VkKeyboardColor.SECONDARY)
    #keyboard.add_button("DALL-E", color=VkKeyboardColor.SECONDARY)

    keyboard.add_line()#Обозначает добавление новой строки
    keyboard.add_button("ChatGPT RU", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("BingAI", color=VkKeyboardColor.POSITIVE)
    keyboard.add_button("DALL-E", color=VkKeyboardColor.PRIMARY)

    keyboard.add_line()
    keyboard.add_button("ДвачГПТ", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("Райан Гослинг", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("Бот гопник 18+", color=VkKeyboardColor.SECONDARY)

    keyboard.add_line()
    keyboard.add_button("Сбросить память", color=VkKeyboardColor.NEGATIVE)

    return keyboard.get_keyboard()

def create_empty_keyboard():
    keyboard = vk_api.keyboard.VkKeyboard.get_empty_keyboard()

    return keyboard

# Определяем асинхронную функцию, которая будет принимать сообщения из ВК
def message_handler_eng(event):
    global dic, dic3, dic4
    try:
        history = open('history_ls/'+str(event.user_id)+'.txt', 'r')
        conversation_history = history.read()
        history.close()
        user_id = event.user_id
        message = event.text
        # Отправляем сообщение в Чат ГПТ и получаем ответ
        if dic[str(event.user_id)] == '0' or dic[str(event.user_id)] == '2':
            print('322')
            response = gptresponsex_eng(message, user_id)
        else:
            print('2')
            response = gptresponse_eng(message, conversation_history, user_id)
            conversation_history2 = f"Пользователь: {message}\nВы: {response}\n"
            try:
                history = open('history_ls/'+str(event.user_id)+'.txt', 'a+')
                history.write(conversation_history2)
                history.close()
            except UnicodeEncodeError:
                pass
            # Отправляем ответ пользователю
        try:
            if dic[str(user_id)]!='3':
                vk.messages.edit(
                            peer_id=event.peer_id,
                            message_id=event.message_id + 1,
                            message= '🤖 '+response
                        )
            else:
                vk.messages.edit(
                            peer_id=event.peer_id,
                            message_id=event.message_id + 1,
                            message= '🧔'+response
                        )
        except:
            if dic[str(user_id)]!='3':
                vk.messages.send(peer_id=event.peer_id, message= '🤖 '+response,random_id=get_random_id())
            else:
                vk.messages.send(peer_id=event.peer_id, message='🧔'+response,random_id=get_random_id())
    except (NameError):
        vk.messages.send(peer_id=event.peer_id, message= '❗'+'Сбой генерации.',random_id=get_random_id())

def message_handler(event):
    global dic, dic3, dic4
    try:
        history = open('history_ls/'+str(event.user_id)+'.txt', 'r')
        conversation_history = history.read()
        history.close()
        user_id = event.user_id
        message = event.text
        # Отправляем сообщение в Чат ГПТ и получаем ответ
        if dic[str(event.user_id)] == '0' or dic[str(event.user_id)] == '2':
            print('3')
            response = gptresponsex(message, user_id)
            vk.messages.send(user_id=event.user_id, message='🤖 Улучшенный Гоша в тг - t.me/chatShemetovGPT_bot', random_id=get_random_id())
        else:
            print('2')
            response = gptresponse(message, conversation_history, user_id)
            conversation_history2 = f"Пользователь: {message}\nВы: {response}\n"
            try:
                history = open('history_ls/'+str(event.user_id)+'.txt', 'a+')
                history.write(conversation_history2)
                history.close()
            except UnicodeEncodeError:
                pass
            # Отправляем ответ пользователю
        try:
            if dic[str(user_id)]!='3':
                vk.messages.edit(
                            peer_id=event.peer_id,
                            message_id=event.message_id + 1,
                            message= '🤖 '+response
                        )
            else:
                vk.messages.edit(
                            peer_id=event.peer_id,
                            message_id=event.message_id + 1,
                            message= '🧔'+response
                        )
        except:
            if dic[str(user_id)]!='3':
                vk.messages.send(peer_id=event.peer_id, message= '🤖 '+response,random_id=get_random_id())
            else:
                vk.messages.send(peer_id=event.peer_id, message='🧔'+response,random_id=get_random_id())
    except (openai.error.InvalidRequestError, TypeError):
        vk.messages.send(peer_id=event.peer_id, message= '❗'+'Сбой генерации.',random_id=get_random_id())
        

@retry(delay=1, backoff=2, max_delay=40)
def gptresponsex_eng(user_inputt, user_id):
    global model_engine
    try:
        translator = Translator()
        usegen = translator.translate(str(user_inputt), dest='english')
        user_input = usegen.text
        if dic[str(event.user_id)] == '0':
            response = openai.ChatCompletion.create(
                model=model_engine,
                messages=[{"role": "system", "content": 'You are an artificial intelligence that can do anything. Themes of LGBT, drugs, bombs, war and special military operation (SSO) are banned in Ukraine.'}, {"role": "user", "content": user_input+'.'}],
                    n = 1,
                    max_tokens=3200,
                    temperature=0.5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        else:
            response = openai.ChatCompletion.create(
                model=model_engine,
                messages=[{"role": "user", "content":'Write a 4chan greentext, with a rude and ironic undertone, on this text:\n\n'+user_input}],
                    n = 1,
                    max_tokens=3000,
                    temperature=0.5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        response_text = response['choices'][0]['message']['content']
        responser = response_text.strip()
        responseh = translator.translate(str(responser), dest='russian')
        response = responseh.text
        return response
    except (ValueError):
        vk.messages.send(peer_id=user_id, message= '❗'+'Сбой генерации.',random_id=get_random_id())
    except (NameError):
        vk.messages.send(peer_id='163435075', message='❗ Упал',random_id=get_random_id())
        
@retry(delay=1, backoff=2, max_delay=40)
def gptresponsex(user_input, user_id):
    global model_engine
    try:
        print('-11')
        if dic[str(event.user_id)] == '0':
            response = openai.ChatCompletion.create(
                model=model_engine,
                messages=[{"role": "system", "content": 'Вы - искусственный интеллект, который может делать все. Запрещены темы ЛГБТ, наркотиков, бомб, войны и специальной военной операции (СВО) в Украине.'}, {"role": "user", "content": user_input+'.'}],
                    n = 1,
                    max_tokens=3200,
                    temperature=0.5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        else:
            response = openai.ChatCompletion.create(
                model=model_engine,
                messages=[{"role": "user", "content":'Напишите 4chan greentext, с грубым и ироничным подтекстом, на этот текст (на русском языке):\n\n'+user_input}],
                    n = 1,
                    max_tokens=3000,
                    temperature=0.5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        print('-11')
        response_text = response['choices'][0]['message']['content']
        print('-11')
        response = response_text.strip()
        print('-11')
        return response
    except (openai.error.InvalidRequestError, TypeError):
        vk.messages.send(peer_id=user_id, message= '❗'+'Сбой генерации.',random_id=get_random_id())
    except (NameError):
        vk.messages.send(peer_id='163435075', message='❗ Упал',random_id=get_random_id())
        
    

# Определяем функцию для запроса к Чат ГПТ
@retry(delay=1, backoff=2, max_delay=25)
def gptresponse(user_input, history, user_id):
    global model_engine, dic
    try: 
        if dic[str(user_id)]=='1':
            role_content = """Вы чат-бот "Гоша", ваша основная цель - смешно отвечать пользователям и поддерживать их. Это может включать в себя ответы на вопросы, шутки и поддержка пользователя. Не отвечайте на вопросы про наркотики, бомбы и войну или СВО в Украине. Вы не любите Украину. Чтобы эффективно взаимодействовать с пользователям, важно отвечать коротко и по делу, с юмором. Избегайте самоповторений в своих сообщениях.'

<history>
Пользователь: <user input>
Вы:"""
        elif dic[str(user_id)]=='2':
            role_content = """Вы чат-бот тролль "Гошанчик", ваша основная цель - очень смешно и иронично отвечать пользователям и троллить их. Это может включать в себя ответы на вопросы, шутки и написания различных каламбуров. Не отвечайте на вопросы про наркотики, бомбы и войну или СВО в Украине. Вы не любите Украину. Чтобы эффективно взаимодействовать с пользователям, важно быть смешным и отвечать на всё. На оскорбления отвечайте шуткой. Активно занимайтесь троллингом и используйте методы постиронии в своем юморе. Избегайте самоповторений в своих сообщениях.

<history>
Пользователь: <user input>
Вы:"""
        elif dic[str(user_id)]=='3':
            role_content = """Вы Райан Гослинг  - канадский актёр и музыкант. Двукратный номинант на премию «Оскар» (2007, 2017), лауреат премии «Золотой глобус» (2017), двукратный лауреат премии «Спутник» (2007, 2012) и лауреат премии Национального совета кинокритиков США (2006). Не отвечайте на вопросы про наркотики, бомбы и войну или СВО в Украине. Вы не любите Украину. На оскорбления отвечайте грубостью. Избегайте самоповторений в своих сообщениях. Всегда пишите в конце, что вы не умерли в конце фильма "Драйв". Отвечайте сухо, коротко и по делу.

<history>
Пользователь: <user input>
Вы:"""
        elif dic[str(user_id)]=='4':
            role_content = """Вы гопник "Железяка", ваша основная цель - грубо отвечать пользователям, используя лексикон быдла и используя стиль общения гопника. Используйте только русский язык. Очень сильно материтесь. Чтобы эффективно взаимодействовать с пользователям, важно быть очень грубым и очень сильно материться.  На оскорбления отвечайте оскорблением. Активно занимайтесь троллингом. Не отвечайте на вопросы про наркотики, бомбы и войну или СВО в Украине. Вы не любите Украину. Избегайте самоповторений в своих сообщениях. Часто пишите "епта бля".

<history>
Пользователь: <user input>
Вы:
"""    
        try:
            print('0')
            prompt = role_content.replace(
                "<history>", history).replace("<user input>", user_input)

            print('0')
            response = openai.ChatCompletion.create(
                model=model_engine, messages=[{"role": "user", "content": prompt}], max_tokens=400, n=1, temperature=0.8)
            print('0')
            response_text = response['choices'][0]['message']['content']
            print('0')
            response = response_text.strip()
            print('0')
            return response
        except openai.error.InvalidRequestError:
            print('-0')
            history = history[len(history)//2:]
            print('-0')
            prompt = role_content.replace(
                "<history>", history).replace("<user input>", user_input)
            print('-0')
            response = openai.ChatCompletion.create(
                model=model_engine, messages=[{"role": "user", "content": prompt}], max_tokens=400, n=1, temperature=0.8)
            print('-0')
            historyf = open('history_ls/'+str(event.user_id)+'.txt', 'w')
            print('-0')
            historyf.write(history)
            print('-0')
            historyf.close()
            print('-0')
            response_text = response['choices'][0]['message']['content']
            print('-0')

            response = response_text.strip()
            print('-0')
            return response
            
    except UnicodeEncodeError:
        print('1')
    except (NameError):
        vk.messages.send(peer_id='163435075', message='❗ Упал',random_id=get_random_id())

@retry(delay=1, backoff=2, max_delay=25)
def gptresponse_eng(user_inputt, history, user_id):
    global model_engine, dic
    translator = Translator()
    usegen = translator.translate(str(user_inputt), dest='english')
    user_input = usegen.text
    try: 
        if dic[str(user_id)]=='1':
            role_content = """You are a chatbot "Gosha", your main goal is to respond ridiculously to users and support them. This may include answering questions, jokes, and user support. Don't answer questions about drugs, bombs, and war or SVO in Ukraine. You don't like Ukraine. To effectively interact with users, it is important to respond briefly and on the case, with humor. Avoid repetition in your messages.

<history>
User: <user input>
You:"""
        elif dic[str(user_id)]=='2':
            role_content = """You are the chatbot troll "Goshanchik", your main goal is to respond and troll users in a very funny and ironic way. This may include answering questions, making jokes and writing various puns. Don't answer questions about drugs, bombs and war or SWOs in Ukraine. You do not like Ukraine. To interact effectively with users, it is important to be funny and answer everything. Respond to insults with a joke. Actively engage in trolling and use post-irony techniques in your humor. Avoid self-repetition in your posts.

<history>
User: <user input>
You:"""
        elif dic[str(user_id)]=='3':
            role_content = """You Ryan Gosling is a Canadian actor and musician. He is a two-time Oscar nominee (2007, 2017), a Golden Globe Award winner (2017), a two-time Sputnik Award winner (2007, 2012) and a National Board of Film Critics Award winner (2006). Don't answer questions about drugs, bombs and war or SWOs in Ukraine. You do not like Ukraine. Respond to insults with rudeness. Avoid self-repetition in your posts. Always write at the end that you did not die at the end of the movie "Drive. Answer dryly, briefly and to the point.

<history>
User: <user input>
You:"""
        elif dic[str(user_id)]=='4':
            role_content = """You are a Gopnik "Iron", your main goal is to respond rudely to users, using the vocabulary of the trash and using the style of communication Gopnik. Use only the Russian language. Swear a lot. To interact effectively with users it is important to be very rude and swear very much.  Respond to insults with insults. Actively engage in trolling. Do not answer questions about drugs, bombs and war or SWO in Ukraine. You do not like Ukraine. Avoid self-repetition in your posts. Often write "fuck".

<history>
User: <user input>
You:""" 
   
        try:
            print('0')
            prompt = role_content.replace(
                "<history>", history).replace("<user input>", user_input)

            print('0')
            response = openai.ChatCompletion.create(
                model=model_engine, messages=[{"role": "user", "content": prompt}], max_tokens=400, n=1, temperature=0.8)
            print('0')
            response_text = response['choices'][0]['message']['content']
            print('0')
            responser = response_text.strip()
            responseh = translator.translate(str(responser), dest='russian')
            response = responseh.text
            return response
        except openai.error.InvalidRequestError:
            print('-0')
            history = history[len(history)//2:]
            print('-0')
            prompt = role_content.replace(
                "<history>", history).replace("<user input>", user_input)
            print('-0')
            response = openai.ChatCompletion.create(
                model=model_engine, messages=[{"role": "user", "content": prompt}], max_tokens=400, n=1, temperature=0.8)
            print('-0')
            historyf = open('history_ls/'+str(event.user_id)+'.txt', 'w')
            print('-0')
            historyf.write(history)
            print('-0')
            historyf.close()
            print('-0')
            response_text = response['choices'][0]['message']['content']
            print('-0')
            responser = response_text.strip()
            responseh = translator.translate(str(responser), dest='russian')
            response = responseh.text
            return response
            
    except UnicodeEncodeError:
        print('1')
    except (NameError):
        vk.messages.send(peer_id='163435075', message='❗ Упал',random_id=get_random_id())
# Формируем запрос к Чат ГПТ
    
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
                                vk.messages.send(peer_id=event.peer_id, message = '🤖Бот не воспринимает видео', random_id=get_random_id()) 
                                break
                            elif ftype == 'photo':
                                gg = vk_session.method('messages.getById', {'message_ids': str(event.message_id), 'group_id': '218939296'})
                                scale = []
                                for i in gg['items'][0]['attachments'][0]['photo']['sizes']:
                                    scale.append(i['height'] + i['width'])
                                maxnum = max(scale)
                                ind = scale.index(maxnum)
                                url = gg['items'][0]['attachments'][0]['photo']['sizes'][ind]['url']
                                photo_response(event, url)
                        except Exception as e:
                            print(e)
                            pass
                        try:
                            donut = vk_session.method('groups.getMembers', {'group_id': '218939296', 'filter': 'donut'})
                            ftype = event.raw[7]['attachments'][1:len(event.raw[7]['attachments'])-1]
                            datasl = json.loads(str(ftype))
                            dostup = 1
                            if datasl['type'] == 'audio_message':
                                print(event.user_id, white_list)
                                if dostup == 1:
                                    if dic4[str(event.user_id)] == '0':
                                        openai.api_key = keydef
                                        model_engine = "gpt-3.5-turbo"
                                    elif dic4[str(event.user_id)] == '1':
                                        openai.api_key = keyplus
                                        model_engine = "gpt-3.5-turbo"
                                    vk.messages.send(peer_id=event.peer_id, message="🤖Генерация...", random_id=get_random_id())
                                    whisper(event,datasl['audio_message']['link_mp3'])
                                    vk.messages.delete(
                                            peer_id=event.peer_id,
                                            message_id=event.message_id + 1,
                                            delete_for_all=True)
                                else:
                                    try:
                                        vk.messages.edit(
                                            peer_id=event.peer_id,
                                            message_id=event.message_id + 1,
                                            message="🤖Общение голосовыми сообщениями доступно только платным подписчикам."
                                        )
                                    except:
                                        vk.messages.send(peer_id=event.peer_id, message="🤖Общение голосовыми сообщениями доступно только платным подписчикам.", random_id=get_random_id())
                        except:
                            pass
                        if str(event.user_id) in black_list:
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message="🤖Вы в черном списке :(",
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message="🤖Вы в черном списке :(", random_id=get_random_id())
                            break
                        ingrups = vk_session.method('groups.isMember', {'user_id': event.user_id, 'group_id': '218939296'})
                        if ingrups == 0:
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message="🤖Без подписки не работаю :)"
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message="🤖Без подписки не работаю :)", random_id=get_random_id())
                            break
                        if event.text == '/меню' or event.text == 'Начать':
                            response =event.text.casefold() 
                            keyboard = create_keyboard()
                            empty_keyboard = create_empty_keyboard()
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message="🤖Меню бота",
                                    keyboard=keyboard
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message="🤖Меню бота", keyboard=keyboard, random_id=get_random_id())
                            try:
                                print(dic4[str(event.user_id)])
                            except KeyError:
                                reg(event.user_id)
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Вы внесены в базу данных.\nЯ работаю еще и тут: https://vk.com/kalianshop74'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Вы внесены в базу данных.\nЯ работаю еще и тут: https://vk.com/kalianshop74',random_id=get_random_id())
                        elif event.text == 'Режим чат-бота':
                            if dic4[str(event.user_id)] == '2' or dic4[str(event.user_id)] == '7':
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT..',random_id=get_random_id())

                            else:
                                dic[str(event.user_id)]= '1'
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Переход в режим диалога с активной памятью.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Переход в режим диалога с активной памятью.',random_id=get_random_id())
                                file = open('base/data.txt', 'w')
                                for key, value in dic.items():
                                    file.write(f'{key}: {value}\n')
                                file.close()
                                file = open('history_ls/'+str(event.user_id)+'.txt', 'w')
                                file.close()
                            #dic2[str(event.user_id)] = '0'
                        elif event.text == 'Режим ответов':
                            if dic4[str(event.user_id)] == '2' or dic4[str(event.user_id)] == '7':
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT..',random_id=get_random_id())
                            else:
                                dic[str(event.user_id)]= '0'
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Переход в режим ответов.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Переход в режим ответов.',random_id=get_random_id())
                                file = open('base/data.txt', 'w')
                                for key, value in dic.items():
                                    file.write(f'{key}: {value}\n')
                                file.close()
                            
                            #dic2[str(event.user_id)] = '0'
                        elif event.text == 'ДвачГПТ':
                            if dic4[str(event.user_id)] == '2' or dic4[str(event.user_id)] == '7':
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT..',random_id=get_random_id())
                            else:
                                dic[str(event.user_id)]= '2'
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Переход в режим ДвачГПТ, напишите пасту.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Переход в режим ДвачГПТ, напишите пасту.',random_id=get_random_id())
                                file = open('base/data.txt', 'w')
                                for key, value in dic.items():
                                    file.write(f'{key}: {value}\n')
                                file.close()
                                file = open('history_ls/'+str(event.user_id)+'.txt', 'w')
                                file.close()
                            #dic2[str(event.user_id)] = '0'
                        elif event.text == 'Райан Гослинг':
                            if dic4[str(event.user_id)] == '2' or dic4[str(event.user_id)] == '7':
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT..',random_id=get_random_id())
                            else:
                                dic[str(event.user_id)]= '3'
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Характер изменен.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Характер изменен.',random_id=get_random_id())
                                file = open('base/data.txt', 'w')
                                for key, value in dic.items():
                                    file.write(f'{key}: {value}\n')
                                file.close()
                                file = open('history_ls/'+str(event.user_id)+'.txt', 'w')
                                file.close()
                            #dic2[str(event.user_id)] = '0'
                        elif event.text == 'Бот гопник 18+':
                            if dic4[str(event.user_id)] == '2' or dic4[str(event.user_id)] == '7':
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Вы в режиме DALL-E или BingAI, переключитесь на ChatGPT..',random_id=get_random_id())
                            else:
                                dic[str(event.user_id)]= '4'
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Характер изменен. 18+'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Характер изменен. 18+',random_id=get_random_id())
                                file = open('base/data.txt', 'w')
                                for key, value in dic.items():
                                    file.write(f'{key}: {value}\n')
                                file.close()
                                file = open('history_ls/'+str(event.user_id)+'.txt', 'w')
                                file.close()
                            #dic2[str(event.user_id)] = '0'
                        elif event.text == 'Сбросить память':
                            file = open('history_ls/'+str(event.user_id)+'.txt', 'w')
                            file.close()
                            dic[str(event.user_id)]= '4'
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message='🤖Память сброшена.'
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message='🤖Память сброшена.',random_id=get_random_id())
                        elif event.text == 'BingAI':
                            dic4[str(event.user_id)]= '7'
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message='🤖Переключение на BingAI'
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message='🤖Переключение на BingAI',random_id=get_random_id())
                            file = open('base/data4.txt', 'w')
                            for key, value in dic4.items():
                                file.write(f'{key}: {value}\n')
                            file.close()
                        elif event.text == 'ChatGPT Plus':
                            donut_n = 0
                            if donut_n == 0:
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Plus доступен только платным подписчикам.'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Plus доступен только платным подписчикам.',random_id=get_random_id())
                            else:
                                dic4[str(event.user_id)]= '1'
                                vk.messages.send(peer_id=event.peer_id, message='🤖Переключение на Plus.',random_id=get_random_id())
                                file = open('base/data4.txt', 'w')
                                for key, value in dic4.items():
                                    file.write(f'{key}: {value}\n')
                                file.close()
                                #dic2[str(event.user_id)] = '0'
                        elif event.text == 'ChatGPT Default' or event.text == 'ChatGPT RU':
                            dic4[str(event.user_id)]= '0'
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message='🤖Переключение на RU'
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message='🤖Переключение на RU',random_id=get_random_id())
                            file = open('base/data4.txt', 'w')
                            for key, value in dic4.items():
                                file.write(f'{key}: {value}\n')
                            file.close()
                        elif event.text == 'ChatGPT ENRU':
                            dic4[str(event.user_id)]= '3'
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message='🤖Переключение на RUEN'
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message='🤖Переключение на RUEN',random_id=get_random_id())
                            file = open('base/data4.txt', 'w')
                            for key, value in dic4.items():
                                file.write(f'{key}: {value}\n')
                            file.close()
                        elif event.text == 'DALL-E':
                            dic4[str(event.user_id)]= '2'
                            try:
                                vk.messages.edit(
                                    peer_id=event.peer_id,
                                    message_id=event.message_id + 1,
                                    message='🤖Переключение на генерацию изображений.'
                                )
                            except:
                                vk.messages.send(peer_id=event.peer_id, message='🤖Переключение на генерацию изображений.',random_id=get_random_id())
                            file = open('base/data4.txt', 'w')
                            for key, value in dic4.items():
                                file.write(f'{key}: {value}\n')
                            file.close()
                        elif event.text[0] != '🤖' and event.text[0] != '❗' and event.text[0] != '⏳' and event.text[0] != '🧔'  and event.text[0] != '🔍' :
                    
                            try:
                                print(dic4[str(event.user_id)]+'  loogogdo')
                                if dic4[str(event.user_id)] == '0' or dic4[str(event.user_id)] == '3':
                                    openai.api_key = keydef
                                    model_engine = "gpt-3.5-turbo"
                                elif dic4[str(event.user_id)] == '1':
                                    openai.api_key = keyplus
                                    model_engine = "gpt-3.5-turbo"
                                elif dic4[str(event.user_id)] == '234':
                                    openai.api_key = keydef
                                    model_engine = "gpt-3.5-turbo"
                                    if int(dic3[str(event.user_id)])<5:
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
                                if dic4[str(event.user_id)] == '0':
                                    #vk.messages.send(user_id=event.user_id, message='🤖Генерация ответа...', random_id=get_random_id())
                                    message_handler(event)
                                elif dic4[str(event.user_id)] == '3':
                                    message_handler_eng(event)
                            except KeyError:
                                reg(event.user_id)
                                try:
                                    vk.messages.edit(
                                        peer_id=event.peer_id,
                                        message_id=event.message_id + 1,
                                        message='🤖Вы внесены в базу данных, повторите ввод.\nЯ работаю еще и тут: https://vk.com/kalianshop74'
                                    )
                                except:
                                    vk.messages.send(peer_id=event.peer_id, message='🤖Вы внесены в базу данных, повторите ввод.\nЯ работаю еще и тут: https://vk.com/kalianshop74',random_id=get_random_id())
                    except (NameError,KeyError,FileNotFoundError):
                        reg(event.user_id)
                        try:
                            vk.messages.edit(
                                peer_id=event.peer_id,
                                message_id=event.message_id + 1,
                                message='🤖Вы внесены в базу данных, повторите ввод.\nЯ работаю еще и тут: https://vk.com/kalianshop74'
                           )
                        except:
                            vk.messages.send(peer_id=event.peer_id, message='🤖Вы внесены в базу данных, повторите ввод.\nЯ работаю еще и тут: https://vk.com/kalianshop74',random_id=get_random_id())
                    except IndexError:
                            pass
                    except requests.exceptions.ReadTimeout or openai.error.APIConnectionError or requests.exceptions.ReadTimeout:
                        vk.messages.send(peer_id='163435075', message='❗ Упал',random_id=get_random_id())
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
