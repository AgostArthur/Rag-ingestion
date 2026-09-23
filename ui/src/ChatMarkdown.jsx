import "katex/dist/katex.min.css";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

function ExternalLink({ node, children, ...props }) {
  return (
    <a {...props} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

function Table({ node, children, ...props }) {
  return (
    <div className="chat-md-table">
      <table {...props}>{children}</table>
    </div>
  );
}

export default function ChatMarkdown({ text }) {
  return (
    <div className="chat-md">
      <ReactMarkdown
        remarkPlugins={[[remarkMath, { singleDollarTextMath: true }], remarkGfm]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: "ignore" }]]}
        components={{ a: ExternalLink, table: Table }}
      >
        {text || ""}
      </ReactMarkdown>
    </div>
  );
}
