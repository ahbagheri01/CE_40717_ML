import pandas as pd
import json
id_type = 0
class category_tree:
    def __init__(self,num):
        global id_type
        self.type = num
        self.id = id_type
        id_type = id_type+1
        self.child = dict()
    def add_child(self,childrens : list,index):
        if len(childrens) <= index:
            return 
        temp = self.child.get(childrens[index],None)
        if temp == None:
            temp = self.child[childrens[index]] = category_tree(self.type+1)
        temp.add_child(childrens,index+1)
    def get_id(self,childrens : list,index,max_depth = 7,depth_iter = 0):
        if len(childrens) <= index or depth_iter+1 > max_depth :
            return self.id
        else:
            return self.child.get(childrens[index]).get_id(childrens,index+1,max_depth,depth_iter+1)
    """
    def __dict__(self):
        temp = ""
        for key,val in self.child.items():
            temp += f'{key} : {val.__dict__}'
            print(temp)
        return f'\{\"type\": {self.type}, \"id\": {self.id}, \"child\": \{{temp}\}\}'
    """
class ConvertorD():
    def __init__(self):
        df = pd.read_csv('utils/category_tree.csv')
        cols = df.to_numpy()
        node = category_tree(0)
        for row in cols:
            node.add_child(row,0)
        self.category_tree = node
        self.map = json.load(open('utils/mapping.json'))
    