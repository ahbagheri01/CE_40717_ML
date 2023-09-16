import json
import pandas as pd
import numpy as np
class preprocess:
    def __init__(self,address = 'utils/mapping.json',addressnum = "utils/numeric_data_mapping.json" ):
        self.map = json.load(open(address))
        self.catcols = ["product_age_group","device_type","product_gender","product_brand","product_country","product_title","audience_id","partner_id","user_id","product_id",]
        self.categorial_mapping = dict()
        self.category_cols = ['product_category(1)',
       'product_category(2)', 'product_category(3)', 'product_category(4)',
       'product_category(5)', 'product_category(6)']
        self.category_mapping = self.map["category_dicts"]
        self.nummap = json.load(open(addressnum))
        self.numcols = ["nb_clicks_1week"]
        self.categorial_col = ["product_age_group","device_type","partner_id", "audience_id",
                  "product_gender","product_category(1)",
                 "product_country","day_time_category"]
        self.numerical_col = ["nb_clicks_1week"]
        self.division = 1
    def fillcat(self,d1,d2,x):
        try:
            return d1[x]
        except:
            return d1['NaN']
    def fill_time_stamp(self,x,division):
        try:
            t = x.split(" ")[1]
            return int(t.split(":")[0])//division + 1 
        except:
            return 0
    def prepro_test(self,d:pd.DataFrame):
        df = self.train_preprocess(d)
        final_df = pd.DataFrame()
        for c in self.categorial_col + self.numerical_col:
            final_df[c] = df[c]
        return final_df.to_numpy()
        
        
    def prepro_train(self,d:pd.DataFrame):
        df = self.train_preprocess(d)
        embed_cols = []
        for c in self.categorial_col:
            embed_cols.append((c,df[c].max()+1))
        cont_cols = ["nb_clicks_1week"]
        target = (df["Sale"].values).astype(int)
        final_df = pd.DataFrame()
        column_idx = {}
        idx = 0
        for c in self.categorial_col + self.numerical_col:
            final_df[c] = df[c]
            column_idx[c] = idx
            idx += 1
        embeddings_input = []
        for i in range(len(embed_cols)):
            val = embed_cols[i][1]
            if embed_cols[i][1] > 100:
                val = 100
            embeddings_input.append((embed_cols[i][0],embed_cols[i][1],val))
        mapping_r= dict()
        mapping_r["column_idx"]  = column_idx
        mapping_r["embeddings_input"]  = embeddings_input
        mapping_r["cont_cols"]  = cont_cols
        json.dump( mapping_r, open( "mapping_r.json", 'w' ) )
        
        return target,final_df.to_numpy(),column_idx,embeddings_input,cont_cols

    
    def train_preprocess(self,d:pd.DataFrame):
        d = d.replace([-1,'-1'],'NaN')
        d = d.replace('0',0)
        for c in self.catcols:
            d[c] = d[c].apply(lambda x : self.fillcat(self.map[c][0],self.map[c][1],x))
        for c in self.category_cols:
            d[c] = d[c].apply(lambda x : self.fillcat(self.category_mapping[c][0],self.category_mapping[c][1],x))
        d["day_time_category"] = d["click_timestamp"].apply(lambda x : self.fill_time_stamp(x,self.division)).astype(int)
        # [upper bound , lowe bound , upper fill , lower fill , null fill, meanfill,stdfill]
        d = d.replace('NaN',np.nan)
        for c in self.numcols:
            m = self.nummap[c]
            mean = m[4]
            d[c] = d[c].fillna(mean)
            upper = m[0]
            lower = m[1]
            d[c] = d[c].apply(lambda x : mean if (x >upper) else lower if (x < lower) else x )
            d[c] = (d[c] - m[5])/m[6] 
        d[self.numcols] = d[self.numcols].astype(float)
        d[self.catcols] = d[self.catcols].astype(int)
        d[self.category_cols] = d[self.category_cols].astype(int)
        return d



