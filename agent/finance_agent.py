class FinanceAgent:
    def __init__(self, monthly_budget):
        self.budget = monthly_budget
        self.spent = 0
    
    def add_expense(self, amount, category):
        self.spent += amount
        remaining = self.budget - self.spent
        
        print(f"Expense: ${amount} ({category})")
        
        if remaining < 0:
            return "ALERT: You have exceeded your budget!"
        elif remaining < (0.2 * self.budget):
            return f"WARNING: Only ${remaining} left. Slow down on {category}."
        else:
            return f"OK. Remaining Balance: ${remaining}"

# Simulation
advisor = FinanceAgent(1000) # $1000 Budget
print("--- Finance Agent ---")
print(advisor.add_expense(200, "Groceries"))
print(advisor.add_expense(700, "Rent"))
print(advisor.add_expense(150, "Dining Out"))