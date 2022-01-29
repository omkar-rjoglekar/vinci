
class hyper_parameters():
    def __init__(self):
        # model
        self.model_name = "model_weight_carrot_50_epochs"
        # Location of the data and name of the file
        self.data_folder = "data/"
        self.image = "train0.npy"
        self.data_location = self.data_folder+self.image
        #reward parameters
        self.max_features = 50
        #ddpg parameters
        self.tau = 0.01
        self.gamma = 0.99
        self.critic_lr = 0.02
        self.actor_lr = 0.01
        self.noise_stddev = 0.2
        self.buffer_size = 50000
        self.total_episodes = 20
        # NN parameters
        self.temperature = 0.1
        self.latent_dim = 256 
        self.input_dimension = 5
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512 
        self.Nz = 128
        self.M = 20
        self.batch_size = 64
        self.max_seq_length = 200


HP = hyper_parameters()
