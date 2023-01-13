import torch

import predictive_coding as pc


class StudentNet(torch.nn.Module):

    def __init__(self, predictive_coding, student_net_hidden_size):
        
        super(StudentNet, self).__init__()
        
        self.predictive_coding = predictive_coding
        
        self.fc1 = torch.nn.Linear(2, student_net_hidden_size, bias=False)
        self.fc2 = torch.nn.Linear(student_net_hidden_size, 1, bias=False)
        
        if self.predictive_coding:
            self.pc1 = pc.PCLayer()

    def forward(self, x):
        
        x = self.fc1(x)
        
        if self.predictive_coding:
            x = self.pc1(x)
        
        x = torch.nn.functional.relu(x)
        
        x = self.fc2(x)
        
        return x
