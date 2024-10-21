#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install transformers')
get_ipython().system('pip install accelerate')
get_ipython().system('pip install bitstandbytes')


# In[7]:


from langchain import PromptTemplate, HuggingFaceHub,LLMChain



# In[16]:


import os
os.environ['HUGGINGFACEHUB_API_TOKEN']="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"


# In[17]:


#text to text generation 
prompt = PromptTemplate(
    input_variable=["product"],
    template="What is the good name for a company that makes {product}"
)


# In[14]:


pip install -U langchain-huggingface


# In[19]:


chain = LLMChain(llm=HuggingFaceHub(repo_id='google/flan-t5-large',model_kwargs={'temperature':0}),prompt=prompt)


# In[20]:


chain.run("camera")


# Text Generation Models | Decoder Only models

# In[23]:


from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline,AutoModelForSeq2SeqLM


# In[21]:


prompt = PromptTemplate(
    input_variables=["name"],
    template="Can you tell me about footballer {name}"
)


# In[26]:


model_id='google/flan-t5-large'


# In[27]:


tokenizer=AutoTokenizer.from_pretrained(model_id)


# In[ ]:


model = AutoModelForSeq2SeqLM.from_pretrained(model_id,device_map="auto")


# In[ ]:


pipeline = pipeline("text2text.generation",model=model,tokenizer=tokenizer,max_length=128)


# In[ ]:


local_llm = HuggingFacePipeline(pipeline=pipeline)


# In[ ]:


prompt = PromptTemplate(
    input_variables=["name"],
    template="Can you tell me about footballer {name}"
)


# In[ ]:


chain = LLMChain(llm=local_llm,prompt=prompt)


# In[ ]:


chain.run("messi")

