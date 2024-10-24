import astropy.table as tb

class chain:
    def __init__(self, filename):
        self.samples=tb.Table.read(filename, format="ascii")
        
        self.post = self.samples["post"]
        self.samples.remove_column("post")
        
        self.weight = self.samples["weight"]
        self.samples.remove_column("weight")
        self.has_wt=True
        
        sep = "END_OF_PRIORS_INI\n"
        text = open(filename).read()
        self.header = text.split(sep)[0]+sep
        self.npar = int(self.header.split("n_varied=")[1].split("\n")[0])
        
        for name in self.samples.dtype.names:
            if name.lower()!=name:
                if name.lower() in self.samples.colnames:
                    print(name.lower)
                else: 
                    self.samples.rename_column(name,name.lower())
             
                
    def add_s8(self, alpha=0.5):
        newcol = self.samples['cosmological_parameters--sigma_8']*((self.samples['cosmological_parameters--omega_m']/0.3)**alpha)
        newcol = tb.Column(newcol, name="cosmological_parameters--s8")
            
        self.samples = tb.Table(self.samples)
        self.samples.add_column(newcol, index=len(self.samples.dtype))
            
        cosmosis_section = 'cosmological_parameters'
        name = 's8'
            
        self.header = self.header.replace("\tpost", "\t%s--%s\tpost"%(cosmosis_section, name))
            
        self.header = self.header.replace("n_varied=%d"%self.npar, "n_varied=%d"%(self.npar+1))
        self.npar+=1