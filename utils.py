def save_checkpoint(checkpoint, file_name = "model_checkpoint.pth.tar") :
    print("==> saving check point ")
    torch.save(checkpoint, file_name)


#TODO
def load_checkpoint(file_name = "model_checkpoint.pth.tar"):
    pass