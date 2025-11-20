class LibraryBot:
    def __init__(self):
        self.books = {"harry potter": "Available", "calculus": "Checked Out", "python ai": "Available"}
        
    def respond(self, user_input):
        user_input = user_input.lower()
        
        if "find" in user_input or "search" in user_input:
            for book in self.books:
                if book in user_input:
                    return f"Book '{book}' is currently {self.books[book]}."
            return "Book not found in catalog."
        
        elif "return" in user_input:
            return "Please place the book in the drop-box at Counter 2."
        
        elif "hours" in user_input:
            return "The library is open from 9 AM to 8 PM."
            
        return "I didn't understand. Try asking about books, returns, or hours."

# Simulation
bot = LibraryBot()
print("--- Library Chatbot ---")
print("User: Find Harry Potter")
print("Bot:", bot.respond("Can you find Harry Potter?"))
print("User: When do you open?")
print("Bot:", bot.respond("What are the hours?"))