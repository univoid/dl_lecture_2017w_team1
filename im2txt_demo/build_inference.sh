cd im2txt
# Build the inference binary.
bazel build -c opt im2txt/run_inference
cd -
pip install pyopenssl
pip install googletrans
pip install pandas
pip install gensim