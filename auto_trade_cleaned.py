#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:18:34 2018

@author: jonathan
"""

from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.opt import Connection


client_id = 105
port = 7496 

def connect(port,clientID):
    tws_conn = Connection.create(port=port, clientId=client_id)
    tws_conn.connect()
    return tws_conn    

con = connect(port=port,clientID=client_id)
con.isConnected()

def messages():
    def error_handler(msg):
        print ("Server Error:", msg)
        
    def server_handler(msg):
        print ("Server Msg:", msg.typeName, "-", msg)    
        
    # Assign error handling function.   # Assign server messages handling function.
    return (con.register(error_handler, 'Error'), con.registerAll(server_handler)) 
    
messages()   

##next valid order ID
con.reqIds(-1)
order_id = 0

#symbol = ticker
#sec_type = 'STK' for stocks
#exch = 'SMART', prim_exch = 'SMART'
#curr = 'USD','EUR' etc
#order_type = 'LMT', 'MKT' etc
#quantity = amount, number of contracts
#actions = 'BUY' or 'SELL'
#time_in_force = 'GTC', 'DAY' etc
#price = price for limit order

def execute (symbol, sec_type, exch, prim_exch, curr,order_type, quantity, action, time_in_force, price):   
    
    global order_id 
    order_id += 1
    
    if sec_type == 'STK':
    
            def create_contract(symbol, sec_type, exch, prim_exch, curr):
            
                contract = Contract()
                contract.m_symbol = symbol
                contract.m_secType = sec_type
                contract.m_exchange = exch
                contract.m_primaryExch = prim_exch
                contract.m_currency = curr
                
                return contract
    
            contract_details = create_contract(symbol, sec_type, exch, prim_exch, curr)
    
    elif sec_type == 'CASH':

            def create_contract(symbol, sec_type, exch, prim_exch, curr):
            
                contract = Contract()
                contract.m_symbol = symbol
                contract.m_secType = sec_type
                contract.m_exchange = exch
                contract.m_primaryExch = prim_exch
                contract.m_currency = curr
                
                return contract   
        
            contract_details = create_contract(symbol, sec_type, exch,prim_exch, curr)
        
    
    def create_order(order_type,quantity, action,time_in_force,price):
        
        order = Order()
        order.m_tif = time_in_force
        order.m_action = action
        order.m_orderType = order_type
        order.m_totalQuantity = quantity
        order.m_lmtPrice = price            
        
        return order
    
    
    order_details = create_order(order_type,quantity, action,time_in_force,price)
    
    con.placeOrder(order_id,contract_details,order_details)

    return None


con.disconnect()
con.connect()

# stock example
execute('TSLA','STK','SMART','SMART','USD','LMT',100,'BUY','GTC',310.00)

# FX example
execute('EUR','CASH','IDEALPRO','IDEALPRO','USD','LMT',50000,'SELL','GTC',1.2163)

## closes all open orders
con.reqGlobalCancel()

## grab all open orders
con.reqAllOpenOrders()




