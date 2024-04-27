# Install requirements.txt
```
pip3 install -r requirements.txt
```

# Run install nltk
```
python3 nltk_install.py
```

# Download data files 
```
mkdir inputs

wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/<part_id>.jsonl.zst

*part_id 2 chu so: 00, 01, ...


sudo apt-get install zstd -y
unzstd <part_id>.jsonl.zst --output-dir-flat inputs
```

# extract and split sentences
```
python3 split_ds.py --path <path.jsonl>
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

## Embeddings
```
    parser.add_argument('--devices', type=str, default="0")
    parser.add_argument("--input_file", type=str, default="data/0.json")
    parser.add_argument("--output_file", type=str, default="output/0/1.npy")

python3 start_embedding.py --devices 0,1,2,3,4,5,6,7 --input_file <input_file> --output_file embeddings/0/1.npy
## Start service
```
pm2 start cache_sevice_v2.py -- --parts 0,1,2 --embeds_dir embeddings/<part> --port 9999  
```

## Start coordinator
```
update validator_hosts.json

{
    "hk1": ["http://localhost:8000"]
}
```

```
pm2 start coordinator_service.py -- --port 9090
```
## Test service
```
curl --location 'http://localhost:8000/texts/distances' \
--header 'Content-Type: application/json' \
--data '{
    "texts": [
        "Almond cookie-like crust, sliced and layered apricots on the bottom, topped with a insanely delicious almond cream, covered in rhubarb ribbons and a crumble topping!"
    ],
    "validator": "hk1"
}'
```