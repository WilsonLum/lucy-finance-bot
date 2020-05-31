def convert(slot):
    stock= {
    			'AAPL' 	: ['Apple', 'apple', 'appl'],
    			'GOOGL'	: ['Google', 'google', 'googl'],
    			'FB'	: ['Facebook', 'facebook', 'fb'],
    			'AMZN'	: ['Amazon', 'amazon', 'amzn'],
    			'MSFT'	: ['Microsoft', 'microsoft','msft'],
    			'BABA'	: ['Alibaba', 'alibaba', 'baba'],
    			'AMD'	: ['amd', 'Amd'],
    			'INTC'	: ['Intel', 'intel', 'intc'],
    			'TSLA'	: ['Tesla', 'tesla', 'tsla'],
    			'DIA'	: ['Dow Jones','Dow', 'dow', 'dow jones', 'dia'],
    			'SPY'	: ['Spider', 'spider', 'spy'],
    			'TWTR'	: ['Twitter', 'twitter', 'twtr']
    		}
    if slot in stock:
        return slot
    else:
        slots = [i for i in stock for j in stock[i] if j == slot]
        return slots[0]
    
 #[stock[i] for i in stock.keys()]


