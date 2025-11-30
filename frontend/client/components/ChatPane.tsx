import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "./ui/button";
import { Send, X, Paperclip, AlertCircle } from "lucide-react";

interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  file: File;
}

interface FileValidationError {
  id: string;
  message: string;
}

export default function ChatPane() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [validationErrors, setValidationErrors] = useState<FileValidationError[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const isValidFileType = (file: File): boolean => {
    const validTypes = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
    const validExtensions = [".txt", ".pdf", ".docx"];

    const hasValidMimeType = validTypes.includes(file.type);
    const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    return hasValidMimeType || hasValidExtension;
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const newFiles: UploadedFile[] = [];
      const errors: FileValidationError[] = [];
      const remainingSlots = 3 - uploadedFiles.length;

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileId = Math.random().toString(36).substr(2, 9);

        if (!isValidFileType(file)) {
          errors.push({
            id: fileId,
            message: `${file.name} - Invalid file type. Only .txt, .pdf, and .docx are allowed.`
          });
        } else if (newFiles.length >= remainingSlots) {
          errors.push({
            id: fileId,
            message: `${file.name} - Maximum 3 files allowed.`
          });
        } else {
          newFiles.push({
            id: fileId,
            name: file.name,
            size: file.size,
            file: file,
          });
        }
      }

      setValidationErrors(errors);
      setUploadedFiles([...uploadedFiles, ...newFiles]);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeFile = (id: string) => {
    setUploadedFiles(uploadedFiles.filter((file) => file.id !== id));
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() && uploadedFiles.length == 0) return;

    const userMessage: Message = {
      id: Math.random().toString(36).substr(2, 9),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages([...messages, userMessage]);
    setInputValue("");
    setValidationErrors([]);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("text", inputValue);

      uploadedFiles.forEach((file) => {
        formData.append("files", file.file);
      });

      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }
  
      const data = await response.json();

      const assistantMessage: Message = {
        id: Math.random().toString(36).substr(2, 9),
        type: "assistant",
        content: data["answer"] || "Response received from the server.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setUploadedFiles([]);
    } catch (error) {
      const errorMessage: Message = {
        id: Math.random().toString(36).substr(2, 9),
        type: "assistant",
        content: `Error: ${error instanceof Error ? error.message : "Failed to send message"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  return (
    <div className="flex flex-col h-full bg-white dark:bg-slate-950">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-slate-200 dark:border-slate-800 px-6 py-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Chat</h2>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Upload documents and ask questions about your graph
        </p>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-slate-500 dark:text-slate-400">
                Start a conversation to build your graph
              </p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.type === "user"
                    ? "bg-blue-600 text-white rounded-br-none"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white rounded-bl-none"
                }`}
              >
                <p className="text-sm">{message.content}</p>
                <p
                  className={`text-xs mt-1 ${
                    message.type === "user" ? "text-blue-100" : "text-slate-500 dark:text-slate-400"
                  }`}
                >
                  {message.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-100 dark:bg-slate-800 px-4 py-2 rounded-lg rounded-bl-none">
              <div className="flex gap-2">
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Validation Errors Section */}
      {validationErrors.length > 0 && (
        <div className="flex-shrink-0 border-t border-red-200 dark:border-red-900 px-6 py-4 bg-red-50 dark:bg-red-900/20">
          <div className="space-y-2">
            {validationErrors.map((error) => (
              <div key={error.id} className="flex items-start gap-2">
                <AlertCircle size={16} className="flex-shrink-0 mt-0.5 text-red-500 dark:text-red-400" />
                <p className="text-sm text-red-700 dark:text-red-300">{error.message}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* File Upload Section */}
      {uploadedFiles.length > 0 && (
        <div className="flex-shrink-0 border-t border-slate-200 dark:border-slate-800 px-6 py-4 bg-slate-50 dark:bg-slate-900">
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-2">
            UPLOADED FILES ({uploadedFiles.length}/3)
          </p>
          <div className="space-y-2">
            {uploadedFiles.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between bg-white dark:bg-slate-800 px-3 py-2 rounded border border-slate-200 dark:border-slate-700"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-900 dark:text-white truncate">
                    {file.name}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {formatFileSize(file.size)}
                  </p>
                </div>
                <button
                  onClick={() => removeFile(file.id)}
                  className="flex-shrink-0 ml-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Section */}
      <div className="flex-shrink-0 border-t border-slate-200 dark:border-slate-800 p-4">
        <div className="space-y-3">
          <div className="flex gap-2">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message... (Shift+Enter for new line)"
              className="flex-1 resize-none rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={3}
            />
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              multiple
              className="hidden"
              accept=".txt,.pdf,.docx"
              disabled={uploadedFiles.length >= 3}
            />
            <Button
              variant="outline"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadedFiles.length >= 3}
              className="flex-shrink-0"
              title={uploadedFiles.length >= 3 ? "Maximum 3 files uploaded" : "Upload document"}
            >
              <Paperclip size={18} />
            </Button>
            
          </div>
          <div className="flex justify-end">
            <Button
              onClick={handleSendMessage}
              disabled={(!inputValue.trim() && uploadedFiles.length == 0) || isLoading}
              className="gap-2"
            >
              <Send size={16} />
              Send
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
