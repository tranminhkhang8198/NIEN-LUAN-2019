import pyttsx3


class Speaker:

    def __init__(self, driverName = 'espeak', language="vi", rate=120, volumn=1.0):
        self.engine = pyttsx3.init(driverName=driverName)
        self.engine.setProperty('voice',language)
        self.engine.setProperty('volumn', volumn)
        self.engine.setProperty('rate',rate)

    def getProperties(self):
        properties = {
            "Language": self.engine.getProperty('voice'),
            "Volumn": self.engine.getProperty('volumn'),
            "Rate": self.engine.getProperty('rate')
        }
        return  properties

    def SetVoice(self, language):
        self.engine.setProperty('voice',language)

    def setVolumn(self, volumn):
        self.engine.setProperty('volumn', volumn)

    def setRate(self, rate):
        self.engine.setProperty('rate', rate)

    def textToSpeech(self, text):

        self.engine.say(text)
        self.engine.runAndWait()
    def freeSpeaker(self):
        self.engine.stop()

