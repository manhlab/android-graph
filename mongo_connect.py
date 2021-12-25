from pymongo import MongoClient
mongo_config = {
    "database_name":"android-graph",
    "collection_name":"result",
    "host":"mongodb+srv://finseed2021:finseed2021@cluster0.lmz9v.mongodb.net/myFirstDatabase?retryWrites=true&w=majority",
    "port":27017
}   

def import2mongo(
    item, database_name: str, collection_name:str, host:str, port: int
):
    mc = MongoClient(host=host, port=port)
    db = mc[database_name]
    collection = db[collection_name]
    collection.update_one({'hash': item['hash']},{"$set": item}, True)
    print(f"Success import {item['hash']} to database!")

def export2mongo(
    item, database_name: str, collection_name:str, host:str, port: int
):
    mc = MongoClient(host=host, port=port)
    db = mc[database_name]
    collection = db[collection_name]
    data = collection.find_one({'hash':item})
    if data:
        return data
    else:
        return None