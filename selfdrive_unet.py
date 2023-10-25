from conv_unet import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
torch.cuda.empty_cache()
#torch.cuda.set_per_process_memory_fraction(0.2)

dataset_dir = "/home/dip_17/Dataset/selfdrive_building"
subfolders = ['dataA', 'dataB', 'dataC', 'dataD', 'dataE']
IMG_SIZE = 512
BATCH_SIZE = 8
images_paths = []
masks_paths = []


for sub in tqdm(subfolders) :
    img_files = sorted(glob.glob(os.path.join(str(dataset_dir + "/" + sub + "/" + sub + "/" + 'CameraRGB') , "*")))
    for file in img_files :
        images_paths.append(file)

    mask_files = sorted(glob.glob(os.path.join(str(dataset_dir + "/" + sub + "/" + sub + "/" + 'CameraSeg') , "*")))
    for file in mask_files :
        masks_paths.append(file)

train_images_dir , test_images_dir , train_masks_dir , test_masks_dir = train_test_split(images_paths , masks_paths , test_size = 0.01)

train_set = CustomDataset(train_images_dir, train_masks_dir)
test_set = CustomDataset(test_images_dir, test_masks_dir)

torch.manual_seed(42)
train_dataloader = DataLoader(
    dataset = train_set ,
    batch_size = BATCH_SIZE ,
    shuffle = True
)
torch.manual_seed(42)
test_dataloader = DataLoader(
    dataset = test_set ,
    batch_size = BATCH_SIZE ,
    shuffle = False
)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

model=UNET(
    in_channels= 3 ,
    out_channels=3
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

epochs = 50
training_loss = []

for i in tqdm(range(epochs)) :
    epoch_loss = 0

    for batch , (image , mask) in enumerate(train_dataloader) :

        image , mask = image.to(device) , mask.to(device)

        mask_pred = model(image)

        loss = criterion(mask_pred , mask)

        if batch % 500 == 0:
            print(f"Looked at {batch * len(image)}/{len(train_dataloader.dataset)} samples.")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss +=loss.item()

    training_loss.append((epoch_loss/len(train_dataloader)))

    print(f"Epoch : {i+1} , Loss: {(epoch_loss/len(train_dataloader))}\n\n")

print(f"The loss of the training set is : {training_loss[-1]}")

test_loss = 0

with torch.no_grad() :

    for image , mask in tqdm(test_dataloader) :

        image , mask = image.to(device) , mask.to(device)

        mask_pred = model(image)

        loss = criterion(mask_pred , mask)

        test_loss += loss

test_loss/=len(test_dataloader)
print(f"The loss of the testing set is : {test_loss}\n")

torch.save(model.state_dict(), 'SegmentCar_building.pt')