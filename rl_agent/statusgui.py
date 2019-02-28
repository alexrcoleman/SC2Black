from tkinter import *
from tkinter.ttk import *

class StatusGUI:
    def __init__(self, root, agents):
      self.root = root
      self.agents = agents
      self.note = Notebook(root, width= 400, height =400)
      self.note.grid()
      root.title("SC2 RL Status")

      self.agent_panels = []
      for i in range(len(agents)):
        panel = AgentPanel(self.note, agents[i])
        self.agent_panels.append(panel)

      self.close_button = Button(root, text="Exit", command=self.close)
      self.close_button.grid()

    def update(self):
      for i in range(len(self.agents)):
        self.agent_panels[i].update()

    def close(self):
      for a in self.agents:
        a.killed = True
      self.root.destroy()

class AgentPanel:
  def __init__(self, root, agent):
    self.agent = agent
    self.root = root
    self.value_text = StringVar()
    self.actions_probs_text = StringVar()
    self.action_text = StringVar()

    tab = Frame(root)
    root.add(tab, text=agent.name)

    value = Label(tab, textvariable=self.value_text)
    action = Label(tab, textvariable=self.action_text)
    action_prob = Label(tab, textvariable=self.actions_probs_text)
    button = Button(tab, text = 'Print agent', command = lambda:print(agent))
    value.grid()
    action.grid()
    action_prob.grid()
    button.grid()

  def update(self):
    self.value_text.set(self.agent.lastValue)
    self.action_text.set(self.agent.lastActionName)
    self.actions_probs_text.set(self.agent.lastActionProbs)
