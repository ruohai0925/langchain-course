import tiktoken

# 创建编码器
encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')

# 输入文本
text = 'Who is the current president of the United States?'

# 计算令牌数
tokens = encoder.encode(text)
print(f'输入文本: \"{text}\"')
print(f'令牌数: {len(tokens)}')
print(f'令牌详情: {tokens}')
print(f'解码验证: {encoder.decode(tokens)}')