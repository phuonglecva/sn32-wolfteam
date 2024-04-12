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

aws s3 cp embeddings/xxx.npy s3://sn32/embeddings/part/xx.npy
```


# Start distance service
## Prepare embeddings datasets
Download dataset from s3 and put them to embeddings/ folder
```
embeddings/
└── 88.npy

0 directories, 1 file
```

create <a><strong>validator_positions.json</strong></a>
```
{
    "hk1": <part_id>
}
```

## Start service
```
pm2 start server.py -- --port 8000
```