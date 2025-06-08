import { 
    Entity,
    PrimaryGeneratedColumn,
    Column,
    BeforeInsert 
} from "typeorm";

import { 
    IsUUID, 
    IsString, 
    IsNumber, 
    IsDate 
} from "class-validator";
import { v4 as uuidv4 } from 'uuid'

@Entity( )
export class Botella {
    @PrimaryGeneratedColumn('uuid')
    @IsUUID()
    id: string;

    @Column()
    @IsString()
    imagen: string;

    @Column()
    @IsString()
    ruta: string;

    @Column()
    @IsString()
    descripcion: string;

    @Column()
    @IsNumber()
    estado: number;

    @Column()
    @IsDate()
    fecha: Date;
    
    @BeforeInsert()
    generateUUID() {
        this.id = uuidv4();
    }
}