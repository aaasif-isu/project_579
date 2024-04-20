# project_579: RAG

Retrieval-augmented generation (RAG) is a framework to give generative models knowledge without finetuning themselves. In this way, an LLM can adapt to new tasks quickly with the presence of new documents.

1. Install packages:
   
    pip install -r requirements.txt

3. Run the following in the terminal:
   
    python main_final.py --pdf_file "paper.pdf"

4. If the script runs successfully, then you will see the following messages:
   
    Connected to database successfully
   
    Indexing successfully

You can see the 1st video demo here: 
https://iastate.box.com/s/0befg4wzngas8i3x45hg6jif2zmzedih


Second part:

1. Run the following in the terminal:
   
    python main_final.py --pdf_file "paper.pdf" --query='what is attention?'

2. Sample output:
   Connected to database
   
   Inserted Successfully
   
   Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.32it/s]


   attention is a mechanism that relates different positions of a single sequence in order to compute a representation of the sequence, and it is a key component in the Transformer, which     is a new type of neural network that can be used for language translation and text summarization without using recurrent or convolutional neural networks. The Transformer uses a series     of layers called sub-layers, each of which performs self-attention, followed by layer normalization. The output embeddings are offset by one position, and masking is used to ensure that    the predictions for position i depend only on the known outputs at positions less than i. Attention can be described as mapping a query and a set of key-value pairs to an output, where     the output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. Multi-      head attention allows the model to jointly attend to information from different representation subspaces at different positions.

You can see the short video for the second milestone here:
https://iastate.box.com/s/leunf877sgdd7t0zcb54nhc526otkw08
