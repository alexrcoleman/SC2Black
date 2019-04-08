from tkinter import *
from tkinter.ttk import *

class StatusGUI:
    def __init__(self, root, envs):
      self.root = root
      self.envs = envs
      self.note = Notebook(root, width= 450, height =270)
      self.note.grid()
      root.title("SC2 RL Status")

      self.agent_panels = []
      for i in range(len(envs)):
        panel = AgentPanel(self.note, envs[i])
        self.agent_panels.append(panel)

      self.close_button = Button(root, text="Exit", command=self.close)
      self.close_button.grid()

    def update(self):
      for i in range(len(self.envs)):
        self.agent_panels[i].update()

    def close(self):
      self.root.destroy()

class AgentPanel:
  def __init__(self, root, env):
    self.env = env
    self.root = root
    self.value_text = StringVar()
    self.actions_probs_text = StringVar()
    self.action_text = StringVar()

    tab = Frame(root)
    root.add(tab, text=env.agent.name)

    value = Label(tab, textvariable=self.value_text)
    action = Label(tab, textvariable=self.action_text)
    action_prob = Label(tab, textvariable=self.actions_probs_text)
    button = Button(tab, text = 'Print agent', command = lambda:print(env.agent))
    value.grid()
    action.grid()
    action_prob.grid()
    button.grid()

  def update(self):
    self.value_text.set(self.env.agent.lastValue)
    self.action_text.set(self.env.agent.lastActionName + " @ " + str(self.env.agent.lastLocation))
    self.actions_probs_text.set(str(self.env.agent.lastActionProbs) + "\n" + str(self.env.agent.last_spatial))
