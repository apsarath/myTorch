echo "=== Downloading LMRD dataset ==="
mkdir -p lmrd_data
cd lmrd_data
wget --continue wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

echo "=== Splitting Validation set from training set ==="
mkdir -p aclImdb/valid
mkdir -p aclImdb/valid/neg
mkdir -p aclImdb/valid/pos
cd aclImdb/train/neg/
mv `ls . | shuf -n 1250 --random-source=<(get_seeded_random 1234) ` ../../valid/neg/
cd ../pos/
mv `ls . | shuf -n 1250 --random-source=<(get_seeded_random 1234) ` ../../valid/pos/
