from gtts import gTTS
from playsound import playsound
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

tts = gTTS(
    text='행복한 기억을 떠올리며 천천히 심호흡을 해보세요. 마음이 안정될거에요.',
    lang='ko', slow=False
)
tts.save('angry.mp3')

tts = gTTS(
    text='나쁜 생각을 버려보세요.',
    lang='ko', slow=False
)
tts.save('disgust.mp3')

tts = gTTS(
    text='마음을 비우기 위해 자보는게 어떨까요? 자고 일어나면 괜찮아질꺼에요.',
    lang='ko', slow=False
)
tts.save('scared.mp3')

tts = gTTS(
    text='당신의 웃음을 보니 또바기도 행복해요',
    lang='ko', slow=False
)
tts.save('happy.mp3')

tts = gTTS(
    text='울고싶을 땐 마음껏 울어보세요, 또바기가 곁에 있어줄게요.',
    lang='ko', slow=False
)
tts.save('sad.mp3')

tts = gTTS(
    text='당신에게 무슨일이 일어나고 있나요? 또바기도 궁금해요!',
    lang='ko', slow=False
)
tts.save('surprised.mp3')

tts = gTTS(
    text='오늘 하루도 웃음으로 시작해보세요. 당신은 웃을 때가 제일 아름다워요.',
    lang='ko', slow=False
)
tts.save('neutral.mp3')

playsound('angry.mp3')