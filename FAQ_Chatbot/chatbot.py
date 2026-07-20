# ==========================================
# CodeAlpha Task 2 - FAQ Chatbot
# Using NLP + Cosine Similarity + Tkinter GUI
# ==========================================

# Import GUI library
from tkinter import *

# Import NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# FAQ Dataset
# ==========================================

faq_data = {

    # Greetings
    "Hello":
        "Hello! How can I help you today?",

    "Hi":
        "Hi there! What can I do for you?",

    "Good morning":
        "Good morning! Hope you're having a great day.",

    "Good afternoon":
        "Good afternoon! How may I assist you?",

    "Good evening":
        "Good evening! How can I help you today?",

    "How are you":
        "I am doing well. Thank you for asking!",

    "What's up":
        "I'm here and ready to help!",

    "Nice to meet you":
        "Nice to meet you too!",

    "Who are you":
        "I am an AI FAQ Chatbot built using Python and NLP.",

    "What is your name":
        "My name is AI FAQ Chatbot.",


    # Company FAQs
    "What are your working hours":
        "We are available from 9 AM to 6 PM Monday to Friday.",

    "Where are you located":
        "We are located in Hyderabad, India.",

    "How can I contact support":
        "You can contact support at support@email.com.",

    "What services do you provide":
        "We provide AI, Machine Learning, Web Development and Software Development services.",

    "Do you offer internships":
        "Yes, we offer internships in multiple domains.",

    "How can I apply for an internship":
        "Visit our careers page and submit your application.",

    "Do you provide certificates":
        "Yes, certificates are provided after successful completion.",

    "Do you offer remote internships":
        "Yes, remote internships are available.",

    "How long is the internship":
        "Internship duration varies from 1 to 3 months.",

    "Is there any fee for internship":
        "Please check the official internship announcement for fee details.",


    # Technical Questions
    "What is Python":
        "Python is a high-level programming language used for web development, AI, data science and automation.",

    "What is machine learning":
        "Machine learning is a branch of AI that enables computers to learn from data.",

    "What is artificial intelligence":
        "Artificial Intelligence is the simulation of human intelligence by machines.",

    "What is deep learning":
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",

    "What is NLP":
        "Natural Language Processing enables computers to understand and process human language.",

    "What is data science":
        "Data science is the field of extracting knowledge and insights from data.",

    "What is a chatbot":
        "A chatbot is a software application that can interact with users through conversation.",

    "What is cloud computing":
        "Cloud computing provides computing services over the internet.",

    "What is cybersecurity":
        "Cybersecurity protects systems, networks and data from digital attacks.",

    "What is an API":
        "An API allows different software applications to communicate with each other.",


    # Education FAQs
    "What is engineering":
        "Engineering is the application of science and mathematics to solve real-world problems.",

    "What is computer science":
        "Computer science is the study of computers, algorithms and software systems.",

    "What is programming":
        "Programming is the process of writing instructions that computers can execute.",

    "What is a database":
        "A database is an organized collection of data.",

    "What is SQL":
        "SQL stands for Structured Query Language and is used to manage databases.",

    "What is Java":
        "Java is a popular object-oriented programming language.",

    "What is C++":
        "C++ is a powerful programming language used in system and application development.",

    "What is Git":
        "Git is a version control system used to track code changes.",

    "What is GitHub":
        "GitHub is a platform used to host and manage Git repositories.",

    "What is VS Code":
        "VS Code is a lightweight and powerful source-code editor developed by Microsoft.",


    # General Questions
    "Tell me a joke":
        "Why do programmers prefer dark mode? Because light attracts bugs!",

    "What is today's date":
        "Please check your device date settings for the current date.",

    "Can you help me":
        "Of course! Ask me any question and I will try my best to help.",

    "Thank you":
        "You're welcome! Happy to help.",

    "Thanks":
        "You're welcome!",

    "Bye":
        "Goodbye! Have a wonderful day.",

    "Goodbye":
        "Goodbye! See you again soon.",

    "See you later":
        "See you later! Take care.",

    "Are you human":
        "No, I am an AI-powered chatbot.",

    "Do you sleep":
        "No, I am available whenever you need assistance."
}

# ==========================================
# Prepare Questions for NLP
# ==========================================

questions = list(faq_data.keys())

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    questions
)


# ==========================================
# Function: Find Best Matching Answer
# ==========================================

def get_response(user_question):

    # Convert user question into vector
    user_vector = vectorizer.transform(
        [user_question]
    )

    # Calculate similarity score
    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )

    # Find best matching question index
    best_match_index = similarity_scores.argmax()

    # Highest similarity value
    confidence_score = similarity_scores[
        0,
        best_match_index
    ]

    # If similarity is too low
    if confidence_score < 0.20:
        return (
            "Sorry, I don't understand that question."
        )

    # Return answer
    return faq_data[
        questions[best_match_index]
    ]


# ==========================================
# Function: Send Message
# ==========================================

def send_message():

    # Get user input
    user_text = user_entry.get().strip()

    # Ignore empty input
    if user_text == "":
        return

    # Display user message
    chat_area.insert(
        END,
        "\nYou: " + user_text + "\n"
    )

    # Get chatbot response
    bot_response = get_response(
        user_text
    )

    # Display chatbot response
    chat_area.insert(
        END,
        "Bot: " + bot_response + "\n"
    )

    # Auto-scroll
    chat_area.see(END)

    # Clear input field
    user_entry.delete(
        0,
        END
    )


# ==========================================
# Function: Clear Chat
# ==========================================

def clear_chat():

    chat_area.delete(
        "1.0",
        END
    )


# ==========================================
# GUI Window
# ==========================================

root = Tk()

root.title(
    "AI FAQ Chatbot"
)

root.geometry(
    "800x600"
)

# Dark Theme
root.configure(
    bg="#1e1e1e"
)


# ==========================================
# Heading
# ==========================================

heading = Label(
    root,
    text="🤖 AI FAQ Chatbot",
    font=("Arial", 20, "bold"),
    bg="#1e1e1e",
    fg="white"
)

heading.pack(
    pady=10
)


# ==========================================
# Chat Display Area
# ==========================================

chat_area = Text(
    root,
    width=90,
    height=25,
    bg="#2d2d2d",
    fg="white",
    font=("Arial", 11)
)

chat_area.pack(
    padx=10,
    pady=10
)

chat_area.insert(
    END,
    "Bot: Hello! Ask me anything.\n"
)


# ==========================================
# Input Frame
# ==========================================

input_frame = Frame(
    root,
    bg="#1e1e1e"
)

input_frame.pack(
    pady=10
)


# ==========================================
# User Input Box
# ==========================================

user_entry = Entry(
    input_frame,
    width=50,
    font=("Arial", 12)
)

user_entry.grid(
    row=0,
    column=0,
    padx=5
)


# ==========================================
# Send Button
# ==========================================

send_button = Button(
    input_frame,
    text="Send",
    command=send_message,
    bg="#0078D7",
    fg="white",
    width=12
)

send_button.grid(
    row=0,
    column=1,
    padx=5
)


# ==========================================
# Clear Button
# ==========================================

clear_button = Button(
    input_frame,
    text="Clear Chat",
    command=clear_chat,
    bg="#DC3545",
    fg="white",
    width=12
)

clear_button.grid(
    row=0,
    column=2,
    padx=5
)


# ==========================================
# Press Enter To Send
# ==========================================

user_entry.bind(
    "<Return>",
    lambda event: send_message()
)


# ==========================================
# Start Application
# ==========================================

root.mainloop()