# RNN Reuters News

(Results and analysis in Wiki)

In this project, different recurrent neural networks will be implemented to analyze the effect that the sequences length has on the classification accuracy of news dataset from the Reuters agency.

The Reuters News dataset contains 11,228 texts corresponding to news classified in 46 categories. The texts are already coded so each word corresponds to an integer. During the data load we specify that we want to work with 2,000 words, so the least repeated words will be considered as unknown.

The Keras Reuters dataset was originally generated by parsing and preprocessing the classic Reuters-21578 dataset. This preprocessing reduced it to 11228 entries, and the provided dataset is missing the label descriptions for the 46 topics.

In similiar format as found in https://martin-thoma.com/nlp-reuters/ for the original 21578 entries Reuters dataset.

Firstly a simple LSTM network is implemented, sencondly a deep LSTM network, which are trained with different text length is performed. Finally, an optimization over previous steps is carried out.
                            
