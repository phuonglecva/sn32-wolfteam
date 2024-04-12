# Download data files 
```
python3 download.py --part part_id
```

# extract and split sentences
```
python3 extract_and_split_ds.py --part part_id
```

# embeddings
```
python3 start_embedding.py --part 88 --devices 0,1,2,3,4,5
```

# upload to s3
```
sudo apt install awscli 
awscli configure (contact admin to get credentials)
```
