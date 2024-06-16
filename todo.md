# TODO
- Train 1M, 3M, 33M without resize the model for both TinyStories, and empty model
    I tried with 33, and 2e-4: It works nice, just it suffers lack of intelligent, I feel it model size. For example when instead of "fahrenheit", I ask for "celsius", still staus "fahrenheit". But the GPTNeo 125, could manage it. Now question isIf I make a let's say 70M model, then train on FineWeb data, instead of TinyStores, can I get the proper response. Or even better still ##M but with FineWeb Data

- Try with Lora for GPT-Neo
- Build my own base model: Pretrain an empty 33M model (same layout) with FineWeb Data, small learning rate, long time, then use it to fine tune for function call

DPO:
When I get the best model, of 33M, hopefully 50M, 100M, and 125M, then I will apply a final DPO as touch up
