class ByteTokenizer:
    pad_id = 256
    bos_id = 257
    eos_id = 258
    def __init__(self):
        pass
    
    def __call__(self, text, max_length = None, padding=True, add_special_tokens = True):
        if type(text)==list:
            return self.batch_encode(text, max_length, padding, add_special_tokens)

        return self.encode(text, max_length, padding, add_special_tokens)
    
    def __len__(self):
        return 256 + 3
    
    def encode(self, text, max_length = None, padding=True, add_special_tokens = True):
        byte_seq = list(text.encode(encoding = 'UTF-8'))
        length = len(byte_seq) + 2

        #truncate if necessary
        if max_length is None:
            max_length = length

        if length > max_length:
            byte_seq = byte_seq[:max_length-2]
            length = max_length
        
        if add_special_tokens:
            return self.add_special_tokens(byte_seq, length, max_length, padding)    

        return byte_seq
    
    def batch_encode(self, texts, max_length = None, padding = True, add_special_tokens = True):
        encodings = []

        for text in texts:
            encodings.append(self.encode(text, max_length, padding, add_special_tokens))
        return encodings
    
    def add_special_tokens(self, encoding, length, max_length, padding):
        
        if padding:
            return [self.bos_id] + encoding + [self.eos_id] + [self.pad_id]*(max_length-length)

        return [self.bos_id] + encoding + [self.eos_id] 
    
    def post_processing(self, out_ids):
        res = []
        for out in out_ids:
            try:
                res.append(out[1:out.index(self.eos_id)])
            except:
                res.append(out)

        return res
    
    def decode(self, byte_seq):
        byte_seqs = self.post_processing([byte_seq])

        return [bytes([item for item in byte_seq if item in range(0,256)]).decode("utf-8", errors="ignore") for byte_seq in byte_seqs]
    
    def batch_decode(self, byte_seqs):

        byte_seqs = self.post_processing(byte_seqs)

        return [bytes([item for item in byte_seq if item in range(0,256)]).decode("utf-8", errors="ignore") for byte_seq in byte_seqs]