import random

def fallback():
	replies = [["Opps, I didnt get you. Can you tell me again?"],
			   ["Sorry, I didn't catch what you're saying."],
			   ["Sorry, I didnt understand. Can you re-phrase?"]]
	return random.choice(replies)
	
