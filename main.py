import os
import speech_recognition as sr
import pyttsx3
from langchain_together import ChatTogether
from langchain.schema import HumanMessage

# Configuration
os.environ["OPENAI_API_KEY"] = "f440501ab89f326cad93347d558f396c0ceb6d0e49131a7a70b1a88cec7b6102"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
TEMPERATURE = 0.7

# Initialize text-to-speech engine
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    # Change voice (0 = male, 1 = female)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    return engine

def initialize_model():
    return ChatTogether(model=MODEL, temperature=TEMPERATURE)

def get_response(llm, prompt):
    system_message = HumanMessage(content=(
        "You are Xae, a friendly assistant for children. "
        "Explain everything in simple words. Use a kind, cheerful tone. "
        "If it's a hard topic, break it into small, fun parts that a kid can understand."
        " give 1-2 line answers."
    ))
    return llm.invoke([system_message, HumanMessage(content=prompt)]).content

def listen_for_keyword(recognizer, keyword):
    with sr.Microphone() as mic:
        while True:
            try:
                recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                audio = recognizer.listen(mic, timeout=None, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                if keyword in text:
                    return True
            except sr.UnknownValueError:
                continue
            except sr.WaitTimeoutError:
                continue
            except sr.RequestError as e:
                print(f"Error with speech service: {e}")
                continue

def get_voice_input(recognizer):
    with sr.Microphone() as mic:
        try:
            recognizer.adjust_for_ambient_noise(mic, duration=0.5)
            audio = recognizer.listen(mic, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio).lower()
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Couldn't understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Speech service error: {e}")
            return ""

def speak_response(engine, text):
    print(f"Xae: {text}\n")
    engine.say(text)
    engine.runAndWait()

def main():
    llm = initialize_model()
    tts_engine = init_tts()
    recognizer = sr.Recognizer()

    print("\n" + "=" * 40)
    print(" Voice-Activated Xae Chatbot 🤖✨ ".center(40, '~'))
    print("=" * 40)

    try:
        while True:
            # Start in sleep mode
            print("\n🔇 System is sleeping. Say 'switch' to activate...")
            tts_engine.say("I'm sleeping. Say switch to wake me up.")
            tts_engine.runAndWait()
            listen_for_keyword(recognizer, "switch")

            print("\n🎤 Voice activated! You can talk to me now.")
            tts_engine.say("Hello! I'm Xae. What would you like to ask?")
            tts_engine.runAndWait()

            # Active listening mode
            while True:
                user_input = get_voice_input(recognizer)

                if not user_input:
                    continue

                if "sleep" in user_input:
                    print("\n🔇 Returning to sleep mode...")
                    tts_engine.say("Okay, going back to sleep. Say switch to wake me again.")
                    tts_engine.runAndWait()
                    break

                if "exit" in user_input or "quit" in user_input:
                    tts_engine.say("Goodbye! Have a great day!")
                    tts_engine.runAndWait()
                    return

                # Generate and speak response
                response = get_response(llm, user_input)
                speak_response(tts_engine, response)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting.")
    finally:
        print("\n" + "=" * 40)
        print(" Session ended ".center(40, '~'))
        print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
