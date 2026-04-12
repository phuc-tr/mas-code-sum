import tree_sitter
from tree_sitter import Language
from tree_sitter import Parser
from typing import Union, Tuple, List



def get_ancestor_type_chains(
        node: tree_sitter.Node
) -> List[str]:
    types = [str(node.type)]
    while node.parent is not None:
        node = node.parent
        types.append(str(node.type))
    return types

class ParserBase:
    def __init__(self,parser_path,language):
        self.lang_object = Language(parser_path, language)
        self.parser = Parser()
        self.parser.set_language(self.lang_object)
        
        
    def parse_code(self,code):
        tree = self.parser.parse(code)
        return tree.root_node

    
    def get_tokens(self, code, root):
        tokens = []
        if len(root.children) == 0 and root.type!="comment":
            #print(root.type)
            tokens.append(code[root.start_byte:root.end_byte].decode())
        else:
            for child in root.children:
                tokens+=(self.get_tokens(code, child))
        return tokens      
    
    def get_token_string(
            self,
            code: str,
            root
    ) -> str:
        tokens = self.get_tokens(code.encode(), root)
        return " ".join(tokens)

    def get_tokens_with_node_type(
            self,
            code: bytes,
            root
    ) -> Tuple[List[str], List[List[str]]]:
        tokens, types = [], []
        if len(root.children) == 0 and root.type=="identifier":
            tokens.append(code[root.start_byte:root.end_byte].decode())
            types.append(get_ancestor_type_chains(root))
        else:
            for child in root.children:
                _tokens, _types = self.get_tokens_with_node_type(code, child)
                tokens += _tokens
                types += _types
        return tokens, types             
    
    
    def dfs_traverse(self, root):
        for child in root.children:
            self.dfs_traverse(child)      

  
base = ParserBase(parser_path="parser/languages.so",language="python")


import json
test_json = []
for line in open('test.jsonl', 'r', encoding="utf-8"):
    test_json.append(json.loads(line))
print(len(test_json)) 
    
    
train_json = []
for line in open('train.jsonl', 'r', encoding="utf-8"):
    train_json.append(json.loads(line))
print(len(train_json)) 
    

with open("test_id_type3.txt","w") as f:
    for i in range(len(test_json)):
        print("###########")
        print(i)
        print("##########")
        code=test_json[i]['code']
        print(code)
        root = base.parse_code(code.encode())
        #print(root)
        tokens,types = base.get_tokens_with_node_type(code.encode(),root)
        print(tokens)
        print(types)
        f.write("We categorized the identifiers into different classes. Please find the information below.\n")
        p="Function name: "+test_json[i]['func_name']
        f.write(p+"\n")
        print(p)
        if len(test_json[i]['func_name'].split("."))>1:
            fname=test_json[i]['func_name'].split(".")[1]
        else:
            fname=test_json[i]['func_name']
        #print(fname)
        
        method_invocation=[]
        formal_parameter=[]
        variable_declarator=[]
        argument_list=[]
        return_type=[]
        access=[]
        
        for j in range(len(tokens)):
            t=tokens[j]
            if t==fname:
                continue
            ty=types[j]
            
            if ty[1]=="parameters":
                if t not in formal_parameter:
                    formal_parameter.append(t)
            elif ty[1]=="return_statement":
                if t not in return_type:
                    return_type.append(t)
            elif ty[1]=="assignment":
                if t not in variable_declarator:
                    variable_declarator.append(t)            
            elif ty[1]=="argument_list":
                if t not in argument_list:
                    argument_list.append(t)                          
            elif ty[1]=="call":
                if t not in method_invocation:
                    method_invocation.append(t)                      
            elif ty[1].find("attribute")!=-1 or ty[1].find("dotted_name")!=-1:
                if t not in access:
                    access.append(t)
        
        if len(formal_parameter)>0:
            f.write("Parameters of the function: ")
            f.write(str(formal_parameter)+"\n")
        if len(return_type)>0:    
            f.write("Identifier to be returned: ")
            f.write(str(return_type)+"\n")
        if len(method_invocation)>0:    
            f.write("Method Invocation: ")
            f.write(str(method_invocation)+"\n")
        if len(argument_list)>0:  
            f.write("Method Arguments: ")
            f.write(str(argument_list)+"\n")  
        if len(variable_declarator)>0:    
            f.write("Assigments: ")
            f.write(str(variable_declarator)+"\n")
        if len(access)>0:    
            f.write("Identifier to access attribute/dotted name: ")
            f.write(str(access)+"\n")        
        
        f.write("##########\n")
  
        


      
with open("train_id_type3.txt","w") as f:
    for i in range(len(train_json)):
        print("###########")
        print(i)
        print("##########")
        code=train_json[i]['code']
        print(code)
        root = base.parse_code(code.encode())
        #print(root)
        tokens,types = base.get_tokens_with_node_type(code.encode(),root)
        print(tokens)
        print(types)
        f.write("We categorized the identifiers into different classes. Please find the information below.\n")
        p="Function name: "+train_json[i]['func_name']
        f.write(p+"\n")
        print(p)
        if len(train_json[i]['func_name'].split("."))>1:
            fname=train_json[i]['func_name'].split(".")[1]
        else:
            fname=train_json[i]['func_name']
        #print(fname)
        
        method_invocation=[]
        formal_parameter=[]
        variable_declarator=[]
        argument_list=[]
        return_type=[]
        access=[]
        
        for j in range(len(tokens)):
           t=tokens[j]
           if t==fname:
               continue
           ty=types[j]
           
           if ty[1]=="parameters":
               if t not in formal_parameter:
                   formal_parameter.append(t)
           elif ty[1]=="return_statement":
               if t not in return_type:
                   return_type.append(t)
           elif ty[1]=="assignment":
               if t not in variable_declarator:
                   variable_declarator.append(t)            
           elif ty[1]=="argument_list":
               if t not in argument_list:
                   argument_list.append(t)                          
           elif ty[1]=="call":
               if t not in method_invocation:
                   method_invocation.append(t)                      
           elif ty[1].find("attribute")!=-1 or ty[1].find("dotted_name")!=-1:
               if t not in access:
                   access.append(t)
       
        if len(formal_parameter)>0:
           f.write("Parameters of the function: ")
           f.write(str(formal_parameter)+"\n")
        if len(return_type)>0:    
           f.write("Identifier to be returned: ")
           f.write(str(return_type)+"\n")
        if len(method_invocation)>0:    
           f.write("Method Invocation: ")
           f.write(str(method_invocation)+"\n")
        if len(argument_list)>0:  
           f.write("Method Arguments: ")
           f.write(str(argument_list)+"\n")  
        if len(variable_declarator)>0:    
           f.write("Assigments: ")
           f.write(str(variable_declarator)+"\n")
        if len(access)>0:    
           f.write("Identifier to access attribute/dotted name: ")
           f.write(str(access)+"\n")        
       
        f.write("##########\n")
 
       

