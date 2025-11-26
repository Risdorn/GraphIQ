import { PanelGroup, Panel, PanelResizeHandle } from "react-resizable-panels";
import ChatPane from "@/components/ChatPane";
import GraphPane from "@/components/GraphPane";

export default function Index() {
  return (
    <div className="w-full h-screen bg-white">
      <PanelGroup direction="horizontal">
        <Panel defaultSize={40} minSize={25}>
          <ChatPane />
        </Panel>
        <PanelResizeHandle className="w-1 bg-slate-200 hover:bg-blue-400 transition-colors cursor-col-resize" />
        <Panel defaultSize={60} minSize={25}>
          <GraphPane />
        </Panel>
      </PanelGroup>
    </div>
  );
}
