# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:25:26 2020

@author: Donal
"""

"""
Design of Context
-----------------
Context of users chat is stored in context.userdata in the following format:
    user_data = context.userdata
    
    user_data["fillslots"] - slots to fill based on previous intent
    
    user_data["slots"] - current state of all slots. 
        To access ticker or days, user_date["slots"]["ticker"] or user_date["slots"]["days"]
        
    user_data["lastintent"] - previous intent name
        This is used to forward the chat back to intents if any slots is detected.
        
"""



