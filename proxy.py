import socket 
import sys
from datetime import datetime

class Proxy:
    # Define the proxy class with required parameters
    def __init__(self, port, timeout, max_object_size, max_cache_size):
        self.port = port
        self.timeout = timeout
        self.maxObjectSize = max_object_size
        self.maxCacheSize = max_cache_size
        self.cache = {}  # Initialize an empty cache
        self.zId = "z5648617"

    # Function to run the proxy
    def runProxy(self):
        host = "127.0.0.1"
        # Create a socket
        socketToClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socketToClient.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow reuse of the address while testing and debugging
        socketToClient.bind((host, self.port))

        # Listen for the connection
        socketToClient.listen(5)
        
        # Listens in a loop
        try:
            while True:
                # Accept a connection
                clientSocket, clientAddress = socketToClient.accept()
                clientSocket.settimeout(self.timeout)

                # Test the connection
                print(f"Received connection from client {clientAddress} at {datetime.now()}")

                # Get request info in the client socket
                self.handleRequest(clientSocket, clientAddress)
                
        # Stop the server using Crtl+C
        except KeyboardInterrupt:
            print("\nServer stopped.")
            sys.exit(0)

    # Function to handle the request from the client
    def handleRequest(self, clientSocket, clientAddress):
        while True:
            # Receive data from the client
            requestData = clientSocket.recv(4096)
            
            # If no data is received, break the loop
            if not requestData:
                print("No data received.")
                break
            
            # Decode the request data
            requestText = requestData.decode('utf-8')
            print(f"This is requestText: {requestText}")

            # Split the request text into lines
            lines = requestText.splitlines()
            # Get the start line of text
            startLine = lines[0]

            # Parse the start line to get the method, URL and HTTP version
            if startLine:
                startLineElements = startLine.split()
                method = startLineElements[0]
                requestTarget = startLineElements[1] 
                protocalVersion = startLineElements[2] 

                print(f"Request Method:{method}")
                print(f"Request URL:{requestTarget}")
                print(f"Http Version:{protocalVersion}")

            # Parse the absolute-form request target
            if requestTarget.startswith("http://"):
                target = requestTarget[7:]
                if '/' in target:
                    hostPort, path = target.split('/', 1)
                    originPath = '/' + path
                else:
                    hostPort = target
                    originPath = '/'

                if ':' in hostPort:
                    host, portStr = hostPort.split(':')
                    port = int(portStr)
                else:
                    host = hostPort
                    # Default port 80
                    port = 80
            
            print(f"Hostname: {host}")
            print(f"Port: {port}")
            print(f"Path: {originPath}")
            
            # Parse the headers
            headers = {}
            for line in lines[1:]:
                # Parse until an empty line
                if line.strip() == "":
                    break
                headerParts = line.split(":", 1)
                if len(headerParts) == 2:
                    headerName = headerParts[0].strip()
                    headerValue = headerParts[1].strip()
                    headers[headerName] = headerValue
            
            print(f"{headerName}:{headerValue}")
            # Parse the request target to get the host and port
            host = ''
            port = 80
            path = '/'
            

            self.requestToServer(method, host, port, path, headers, clientSocket)

            # Gnerate the log

            # Only GET method has cache status of cache hit or miss
            if method == "GET":
                cacheStatus = "H"
            else:
                cacheStatus = "-"

            self.generateLog(clientAddress, cacheStatus, startLine, "200", len(requestData))
        return requestData

    # Function to generate request to the server
    def requestToServer(self, method, host, port, path, headers, clientSocket):
        # Create a socket to the server
        socketToServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socketToServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow reuse of the address while testing and debugging
        socketToServer.bind((host, port))

        # Generate a request to the server
        requestLine = f"{method} {path} HTTP/1.1\r\n"
        headers["Host"] = host
        headers["Connection"] = "close"
        headers.pop("Proxy-Connection", None)
        via = f"1.1 {self.zId}"
        if "Via" in headers:
            headers["Via"] += f", {via}"
        else:
            headers["Via"] = via

        headerText = ''.join(f"{k}: {v}\r\n" for k, v in headers.items())
        fullRequest = (requestLine + headerText + "\r\n").encode('utf-8')
        socketToServer.sendall(fullRequest)

        # Receive response
        response = b''
        while True:
            data = socketToServer.recv(4096)
            if not data:
                break
            response += data
            clientSocket.sendall(data)

    # Function to generate the log
    def generateLog(self, clientAddress, cacheStatus, request, status, bodySize):
        # Log format with the following syntax:
        # host port cache date request status bytes
        clientIp, clientPort = clientAddress
        currentTime = datetime.now().astimezone().strftime("[%d/%b/%Y:%H:%M:%S %z]")
        log = f"{clientIp} {clientPort} {cacheStatus} {currentTime} \"{request}\" {status} {bodySize}\n"
        print(log)
      
def parseCommandLine():
    # Parse command line arguments
    if len(sys.argv) != 5:
        print("Usage: python proxy.py <port> <timeout> <max_object_size> <max_cache_size>")
        sys.exit(1)
    
    try:
        port = int(sys.argv[1])
        timeout = int(sys.argv[2])
        max_object_size = int(sys.argv[3])
        max_cache_size = int(sys.argv[4])

    except ValueError:
        print("Error: All arguments must be integers.")
        sys.exit(1)

    # Validate port number
    if port < 49152 or port > 65535:
        print("Warning: Port number between 49152 and 65535 is recommended.")
        sys.exit(1)

    # Validate size
    if max_cache_size < max_object_size:
            print("Warning: max_cache_size must be greater than or equal to max_object_size.")
            sys.exit(1)

    return port, timeout, max_object_size, max_cache_size

# Main entrance for command line arguments
def main(): 
        # Parse command line
        port, timeout, max_object_size, max_cache_size = parseCommandLine()

        # Run the proxy
        proxy = Proxy(port, timeout, max_object_size, max_cache_size)
        proxy.runProxy()

# Run main function
main()