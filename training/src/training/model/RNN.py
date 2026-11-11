
import torch 
import torch.nn as nn 
class RNNForecaster(nn.Module):
    def __init__(self,input_size=6,hidden_size=32,num_layers=1,output_size=1,dropout=0.0,nonlinearity:str="tanh")->None:
        super(RNNForecaster,self).__init__()
        if input_size<=0 or hidden_size<=0:
            raise ValueError("input_size and hidden must be positive")
        if num_layers<=0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <=dropout<=1.0:
            raise ValueError("dropout must be in [0,1)")

        self.rnn=nn.RNN(input_size,hidden_size,num_layers=num_layers,nonlinearity=nonlinearity,batch_first=True,dropout=dropout if num_layers>1 else 0.0)

        self.output=nn.Linear(hidden_size,output_size)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        output,hidden=self.rnn(x)
        #Output: h1,h2,...,ht 
        #Aqui esta el [batch,sequence_length,hidden_size]
        last_hidden=output[:,-1,:]
        prediction =self.output(last_hidden)
        return prediction.squeeze(-1)
