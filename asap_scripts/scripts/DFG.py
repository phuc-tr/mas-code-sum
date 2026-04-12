from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

import argparse



def process(data_folder,lang):
    
    #load parsers
    parsers={}        
    for lang in dfg_function:
        LANGUAGE = Language('parser/my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE) 
        parser = [parser,dfg_function[lang]]    
        parsers[lang]= parser
        
        
    #remove comments, tokenize code and extract dataflow                                        
    def extract_dataflow(code, parser,lang):
        #remove comments
        try:
            code=remove_comments_and_docstrings(code,lang)
        except:
            pass    
        #obtain dataflow
        if lang=="php":
            code="<?php"+code+"?>"    
        try:
            tree = parser[0].parse(bytes(code,'utf8'))    
            root_node = tree.root_node  
            tokens_index=tree_to_token_index(root_node)     
            code=code.split('\n')
            code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
            index_to_code={}
            for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
                index_to_code[index]=(idx,code)  
            try:
                DFG,_=parser[1](root_node,index_to_code,{}) 
            except:
                DFG=[]
            DFG=sorted(DFG,key=lambda x:x[1])
            indexs=set()
            for d in DFG:
                if len(d[-1])!=0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG=[]
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg=new_DFG
        except:
            dfg=[]
        return code_tokens,dfg
    
    
    
    import json
    test_json = []
    for line in open(data_folder+'/test.jsonl', 'r', encoding="utf-8"):
        test_json.append(json.loads(line))
    print(len(test_json)) 
        
        
    train_json = []
    for line in open(data_folder+'/train.jsonl', 'r', encoding="utf-8"):
        train_json.append(json.loads(line))
    print(len(train_json))
        
    
    
    with open(data_folder+"/test_dfg.txt","w") as f:
        for i in range(len(test_json)):
            flag=0
            print("###########")
            print(i)
            print("##########")
            #code=test_json[i]['original_string']
            code_tokens,dfg=extract_dataflow(test_json[i]['original_string'],parser,lang)
            f.write("Please find the dataflow of the function. We present the source and list of target indices.\n" )
            
            dictionary={}
            
            for w in dfg:
                if len(w[4])==0:
                    continue
                flag=1
                key=w[0]+"-"+str(w[4])
                if key not in dictionary:
                    dictionary[key]=[]
                    dictionary[key].append(w[1])
                else:
                    dictionary[key].append(w[1])
            count=0
            for key in dictionary:
                f.write(key+" "+str(dictionary[key])+"\n")
                count=count+1
                if count==30:
                    break
            if flag==0:
                f.write("No DFG available\n")
            f.write("##########\n")
            
            
    with open(data_folder+"/train_dfg.txt","w") as f:
        for i in range(len(train_json)):
            flag=0
            print("###########")
            print(i)
            print("##########")
            #code=train_json[i]['original_string']
            code_tokens,dfg=extract_dataflow(train_json[i]['original_string'],parser,lang)
            f.write("Please find the dataflow of the function. We present the source and list of target indices.\n" )
            
            dictionary={}
            
            for w in dfg:
                if len(w[4])==0:
                    continue
                flag=1
                key=w[0]+"-"+str(w[4])
                if key not in dictionary:
                    dictionary[key]=[]
                    dictionary[key].append(w[1])
                else:
                    dictionary[key].append(w[1])
            count=0
            for key in dictionary:
                f.write(key+" "+str(dictionary[key])+"\n")
                count=count+1
                if count==30:
                    break
            if flag==0:
                f.write("No DFG available\n")  
            f.write("##########\n")
        

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="data folder path ")
    parser.add_argument("--language", default=None, type=str, required=True,
                        help="csharp/cpp")
    args = parser.parse_args()
    
    
    process(args.data_folder,args.language)
    

main()    