import runpod
import time


def handler(event):

    print(f"Worker Start")
    input = event['input']

    prompt = input.get('prompt')
    seconds = input.get('seconds', 0)

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")

    # You can replace this sleep call with your own Python code
    #time.sleep(seconds)

    from run_csm import main
    main()



    return prompt


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})