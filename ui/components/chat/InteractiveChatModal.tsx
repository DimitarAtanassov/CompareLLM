// components/chat/InteractiveChatModal.tsx
import React, { useRef, useEffect } from 'react';
import { ModelChat, ChatMessage } from '../../types';

interface InteractiveChatModalProps {
  activeModel: string | null;
  modelChats: Record<string, ModelChat>;
  interactivePrompt: string;
  onInteractivePromptChange: (prompt: string) => void;
  onSendMessage: () => void;
  onCloseChat: () => void;
  originalPrompt: string;
}

export const InteractiveChatModal: React.FC<InteractiveChatModalProps> = ({
  activeModel,
  modelChats,
  interactivePrompt,
  onInteractivePromptChange,
  onSendMessage,
  onCloseChat,
  originalPrompt
}) => {
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [modelChats, activeModel]);

  if (!activeModel) return null;

  const currentChat = modelChats[activeModel];
  if (!currentChat) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl w-full max-w-4xl h-[80vh] max-h-[800px] flex flex-col border border-zinc-200 dark:border-zinc-700">
        {/* Chat Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
          <h2 className="text-lg font-semibold">
            Chat with <span className="font-mono text-orange-600 dark:text-orange-400">{activeModel}</span>
          </h2>
          <button
            onClick={onCloseChat}
            className="p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800 transition"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {currentChat.messages.map((message, index) => {
            const isFromOriginalRun = index <= 1 && originalPrompt.trim() && message.content === originalPrompt.trim();
            const isOriginalResponse = index === 1 && currentChat.messages[0]?.content === originalPrompt.trim();
            
            return (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className="max-w-[80%] space-y-1">
                  {(isFromOriginalRun || isOriginalResponse) && (
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 px-2">
                      From comparison run
                    </div>
                  )}
                  <div
                    className={`rounded-2xl px-4 py-2 ${
                      message.role === "user"
                        ? "bg-orange-600 text-white"
                        : "bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
                    }`}
                  >
                    <pre className="whitespace-pre-wrap text-sm">{message.content}</pre>
                    <div className="text-xs opacity-70 mt-1">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          {currentChat.isStreaming && currentChat.currentResponse && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-2xl px-4 py-2 bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100">
                <pre className="whitespace-pre-wrap text-sm">{currentChat.currentResponse}</pre>
                <div className="text-xs opacity-70 mt-1">Typing...</div>
              </div>
            </div>
          )}

          <div ref={chatBottomRef} />
        </div>

        {/* Chat Input */}
        <div className="p-4 border-t border-zinc-200 dark:border-zinc-700">
          <div className="flex gap-3">
            <textarea
              value={interactivePrompt}
              onChange={(e) => onInteractivePromptChange(e.target.value)}
              placeholder="Type your message..."
              rows={2}
              className="flex-1 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-800 resize-none"
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  onSendMessage();
                }
              }}
            />
            <button
              onClick={onSendMessage}
              disabled={!interactivePrompt.trim() || currentChat.isStreaming}
              className="px-6 py-2 rounded-xl font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
            >
              Send
            </button>
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
            Press Cmd/Ctrl + Enter to send • Esc to close
          </div>
        </div>
      </div>
    </div>
  );
};
