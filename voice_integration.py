import os
import sys
import time

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from google.cloud import speech, texttospeech
import pyaudio
import pygame


DATA_PATH = "courses_data/"
VECTOR_STORE_PATH = "vector_store"
RATE = 16000
CHUNK = int(RATE / 10)


def generate_audio_chunks():
    """Generator that yields audio chunks from the microphone."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Listening... (Speak now)")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            yield data
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def listen_print_loop(responses):
    """Iterates through server responses and prints them."""
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if result.is_final:
            sys.stdout.write(f"\rFinal Transcript: {transcript}\n")
            sys.stdout.flush()
            return transcript
        else:
            sys.stdout.write(f"\rPartial: {transcript} ")
            sys.stdout.flush()
    return ""

def get_stt():
    """Gets the user's speech and returns the final transcript."""
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",  # Changed to en-US for general use
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    audio_generator = generate_audio_chunks()
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)

    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )

    transcript = listen_print_loop(responses)
    return transcript

# --- 3. Text-to-Speech (TTS) Function ---

def get_tts(output_text):
    """Synthesizes speech from the input string of text."""
    try:
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=output_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            name="en-US-Wavenet-D"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        tts_response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open("response.wav", "wb") as out:
            out.write(tts_response.audio_content)
        
        print("Bot is speaking...")
        pygame.mixer.init(frequency=RATE)
        pygame.mixer.music.load('response.wav')
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    except Exception as e:
        print(f"Error in TTS: {e}")
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

def get_conversational_rag_chain(retriever):
    """
    Creates the main conversational RAG chain.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "You need to talk like an support team member talks on call regarding queries , take the retrieved content as context and talk as humanly as possible"
        "Your output will be given to a speech client so do not include any *,symbols just plain text that a person acna speak "
        "without the chat history. Do NOT answer the question, "
        "answer in less than 20 words max dont go over that - dont over explain things "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are a helpful student support assistant for a university. "
        "Use the following retrieved context to answer the user's question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep your answers concise and helpful.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def main():
    """Main function to run the voice-enabled RAG bot."""

    print("Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    conversational_rag_chain = get_conversational_rag_chain(retriever)

    print("\n----------------------------------------------------")
    print("Voice Student Support Bot is ready!")
    print("Say 'exit' or 'quit' to end the chat.")
    print("----------------------------------------------------")

    chat_history = []

    while True:
        try:
            user_input = get_stt()

            if not user_input:
                print("No input received. Please try again.")
                continue

            if user_input.lower() in ['exit', 'quit']:
                print(" Goodbye!")
                get_tts("Goodbye!")
                break

            print("Thinking...")
            response = conversational_rag_chain.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
            bot_response = response['answer']
            print(f"Bot: {bot_response}")

            get_tts(bot_response)
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=bot_response))
            
            time.sleep(1) 

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()