import socket
import mimetypes

# Create a sockt
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the local host and port, dynamic port range from49152 to 65535
host = "127.0.0.1"
port = 55555
serverSocket.bind((host, port))

# Listen for the connection
serverSocket.listen(1)

# The server listens in a loop
while True:
    # Accept the connection
    connectionSocket, clientAddress = serverSocket.accept()

    # Test connection
    print(f"Received connection from client {clientAddress}")

    keep_alive = True
    while keep_alive:
        try:
            # Get request information in the client socket
            requestData = connectionSocket.recv(1024)
            if not requestData:
                break
            print(f"==============================================")
            print(f"This is requestData from client: {requestData}")
            requestText = requestData.decode()
            print(f"==============================================")
            print(f"This is requestText after decode: {requestText}")

            # See the first line of requestText, if it contains GET request and file name requested
            firstLine = requestText.splitlines()[0]

            # Split GET and file name
            firstLineElements = firstLine.split() 
            if firstLineElements[0] == "GET":
                # Remove slash before the file name
                filename = firstLineElements[1].lstrip("/")
            else:
                # This is not a GET request, the server only processes GET requests
                print(f"This is not a GET requset!")
                break

            # Get the requested file and read it
            try:
                file = open(filename, "rb")
                print(f"This is file: {file}")
                fileContent = file.read()
                print(f"This is file content: {fileContent}")
                isFile = True
            except FileNotFoundError:
                # If the filename does not exist (when FileNotFoundError occurs)
                fileContent = b"<h1>404 Not Found</h1>"
                isFile = False

            # Check if client wants to close the connection
            if "Connection: close" in requestText or "connection: close" in requestText:
                connection_header = "close"
                keep_alive = False
            else:
                connection_header = "keep-alive"

            # Create HTTP response with header lines
            # The content-type should match the file type, or it will cause messy code for the browser
            if isFile:
                contentType = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                print(f"This is contentType: {contentType}")
                responseHeader = f"HTTP/1.1 200 OK\r\nContent-Type: {contentType}\r\nContent-Length: {len(fileContent)}\r\nConnection: {connection_header}\r\n\r\n"
            else:
                responseHeader = f"HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\nContent-Length: {len(fileContent)}\r\nConnection: {connection_header}\r\n\r\n"

            response = responseHeader.encode() + fileContent

            # Send the response to the browser
            connectionSocket.sendall(response)

        except Exception as e:
            print(f"Exception occurred: {e}")
            break

    connectionSocket.close()