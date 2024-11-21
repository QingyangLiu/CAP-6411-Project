import yaml
import argparse
from src.models.model_loader import CfgLoader
from src.models.model_loader import ModelLoader
from importlib import import_module
from src.datasets.nuscenes_qa import NuScenes_QA
import torch
import numpy as np
from flask import Flask, request, render_template_string
import io
from PIL import Image

app = Flask(__name__)

# HTML template for the file upload form
UPLOAD_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload Files and Text</title>
</head>
<body>
    <h1>Upload 6 Files and Enter a Text String</h1>
    <form method="POST" enctype="multipart/form-data">
        <p><input type="text" name="text_input" placeholder="Enter your text here" required></p>
        <p><input type="file" name="file1" required></p>
        <p><input type="file" name="file2" required></p>
        <p><input type="file" name="file3" required></p>
        <p><input type="file" name="file4" required></p>
        <p><input type="file" name="file5" required></p>
        <p><input type="file" name="file6" required></p>
        <p><button type="submit">Submit</button></p>
    </form>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def upload_files_and_text():
    if request.method == "POST":
        # Retrieve the text input
        text_input = request.form.get("text_input")

        # Collect all uploaded files
        uploaded_files = [
            request.files.get(f"file{i}") for i in range(1, 7)
        ]
        
        # Ensure all files are uploaded and text input is provided
        if None in uploaded_files or not text_input:
            return "Please upload all 6 files and provide a text input.", 400
        
        # Open the images using Pillow and print their sizes
        image_sizes = []
        for file in uploaded_files:
            try:
                # Open the image
                image = Image.open(file)
          #      image.show()
                image_sizes.append(image.size)  # Get size (width, height)
            except Exception as e:
                return f"Error opening image: {file.filename}. Error: {str(e)}", 400

        ques_embed=torch.from_numpy(dataset.proc_ques(text_input, 30))
        ques_embed=ques_embed.unsqueeze(0)

        obj_embed=torch.randn(1, 300, 512)
        bbox_embed=torch.randn(1, 300, 7)


        pred=runModel(pre_net, net, ques_embed, obj_embed, bbox_embed)

        return_text=dataset.ix2ans[str(pred[0])]
        


        # Return the image sizes and text input to the client
        #return f"Files uploaded successfully! Your text input was: {return_text}. Image sizes: {image_sizes}"
        return f"Response: {return_text}"
    # Render the upload form
    return render_template_string(UPLOAD_FORM)



def runModel(pre_net, net, ques_embed, obj_embed, bbox_embed):
        obj_B, obj_N, obj_C=obj_embed.shape  
        bbox_B, bbox_N, bbox_C=bbox_embed.shape  

        obj_embed=torch.reshape(obj_embed, (int(obj_B*obj_N/100), 100, obj_C))
        bbox_embed=torch.reshape(bbox_embed, (int(bbox_B*bbox_N/100), 100, bbox_C))
        ques_embed_temp=ques_embed.repeat_interleave(3, dim=0)
              
        __C.USE_BBOX_FEAT=True
        aggregate=pre_net(obj_embed, bbox_embed, ques_embed_temp, return_norm_only=True)
                
        aggregate=torch.reshape(aggregate, (obj_B, 3, -1))

        __C.USE_BBOX_FEAT=False

        pred = net(
            aggregate,
            bbox_embed,
            ques_embed
        )
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)

        return pred_argmax




def CreateModel():
    cfg_file = 'configs/{}.yaml'.format('mcan_small')
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    __C.MODEL_USE='mcan'
    __C.RUN_MODE='val'
    __C.VIS_FEAT='CenterPoint'
 

    dataset = NuScenes_QA(__C)
    pretrained_emb=dataset.pretrained_emb
    token_size=70
    ans_size=30
    __C.USE_BBOX_FEAT=True
    pre_net=ModelLoader(__C).Net(__C, pretrained_emb, token_size, ans_size)

    ckpt=torch.load('/home/qi940700/Desktop/NuScenes-QA-new-v2/epoch13.pkl')
    pre_net.load_state_dict(ckpt['state_dict'])


    pre_net=pre_net.eval()
    __C.USE_BBOX_FEAT=False
    __C.FEAT_SIZE['OBJ_FEAT_SIZE']=(100, 1024)


    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )
    ckpt=torch.load('/home/qi940700/Desktop/NuScenes-QA-new-v2/outputs/ckpts/ckpt_8123330/epoch11.pkl')
    net.load_state_dict(ckpt['state_dict'])
    return pre_net, net, dataset, __C

pre_net, net, dataset, __C=CreateModel()











if __name__ == '__main__':
    app.run(debug=True)

    ques_embed=torch.from_numpy(dataset.proc_ques('hello, my name is bob', 30))
    ques_embed=ques_embed.unsqueeze(0)

    obj_embed=torch.randn(1, 300, 512)
    bbox_embed=torch.randn(1, 300, 7)

    pred=runModel(pre_net, net, ques_embed, obj_embed, bbox_embed)

    print(dataset.ix2ans[str(pred[0])])







