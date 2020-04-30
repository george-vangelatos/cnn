def Batcher(n, batch_size): # partitions n items into batches of batch_size items (or whatever is left at the end)
    for start in range(0, n, batch_size): # loop over each start value
        num = min(batch_size, n - start) # work out size of current batch
        yield start, num # yield the current batch




