export FILEID=1AehHWRJgDQDmiTOiHFlVHIjwlkHOy5ME 
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O source.zip && rm -rf /tmp/cookies. txt
unzip source.zip && rm source.zip
wget https://rocketqa.bj.bcebos.com/corpus/nq.tar.gz
tar -zxvf 'nq.tar.gz' nq/para.txt nq/para.title.txt && mv nq/para.txt nq/para.title.txt source/ && rm -rf nq nq.tar.gz
pip install -r requirements.txt
mkdir qa prior post