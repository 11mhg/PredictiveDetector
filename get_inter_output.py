import os
import argparse
from tracker import *
from PIL import Image

# Args
argparser = argparse.ArgumentParser(
            description="Get Intermediary Output from Tracker network")

argparser.add_argument(
            '-l',
                '--output_layer',
                    help="Output layer to retrieve (cannot be less than 1 or more than the size of the network.",
                        default=3)



def main(args):
    img_dir = './Dataset/MOT/images/train/MOT17-02/img1/'
    t = Tracker()
    t.model_body.load_weights('tracker_weights_best_so_far.h5')
    t.loaded = True

    output_ind = int(args.output_layer)
    
    model = Model(inputs=t.model_body.input, outputs=t.model_body.layers[output_ind].output)

    model.summary()

    answer = input("Does this look okay? (y or n)")

    if "n" in answer:
        return

    frames = [1,2,3,4,5]

    filenames = [img_dir+str(i).zfill(6)+'.jpg' for i in frames]

    image_data_init = [Image.open(i) for i in filenames]
    image_data = [i.resize((608,608),Image.BICUBIC) for i in image_data_init]
    image_data = [np.array(i,dtype='float32')/255. for i in image_data]
    image_data = np.expand_dims(image_data,0)

    out = model.predict(image_data)

    out_new = []
    for i in range(out.shape[3]):
        im = Image.fromarray(out[0,:,:,i]*255)
        im = im.convert("RGB")
        im.save('./Intermediary_Outputs/{}_at_output_layer_{}.jpg'.format(i,output_ind))
    print("Done")










if __name__== '__main__':
    args = argparser.parse_args()
    main(args)
